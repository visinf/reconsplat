'''
    Custom loader for DL3DV dataset. Same structure as the loader from DepthSplat, 
    with support for loading depth labels and other cameras from a different source (e.g., VGGT).
'''

import itertools
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import torch
import torchvision.transforms as tf
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import numpy as np

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon, SHARD_SHUFFLE_SEED
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.depth_io import load_scene_from_zarr
from .label_paths import resolve_label_dir

# Resolution the processed labels (VGGT depth and cameras) were produced at. Images are brought to
# it so the pixel-valued VGGT intrinsics line up, and skip_bad_shape checks against it below.
LABEL_RESOLUTION = (252, 448)
import os
import re

# Train scenes to skip.
SKIP_SCENES = [
    'd039dadf4a29e0cb81ace31443d69c85d75d686c87605359f101ec3684c5bcd2',
    'f5998f682e38b5135060825a11b73b46ba3e2f668fa92e5d3885757ac799aac2',
    'dl3dv_d039dadf4a29e0cb81ace31443d69c85d75d686c87605359f101ec3684c5bcd2',
    'dl3dv_f5998f682e38b5135060825a11b73b46ba3e2f668fa92e5d3885757ac799aac2'
]

@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    roots: list[Path]
    labels_root: Path | None
    load_depth_labels: bool
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    ori_image_shape: list[int]
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    shuffle_val: bool = True
    no_mix_test_set: bool = True
    load_depth: bool = False
    min_views: int = 0
    max_views: int = 0
    sort_target_index: Optional[bool] = False
    overfit_max_views: Optional[int] = None
    sort_context_index: Optional[bool] = False
    use_index_to_load_chunk: Optional[bool] = False
    load_labels_from_zarr_dirs: bool = True
    load_other_cameras: bool = True
    cameras_root: Path | None = None
    # Per-scene multiplier on the camera translations, as {scene: scale}; same field and format as
    # DatasetRE10kCfg. Needed when evaluating on the chunks' own (COLMAP) poses, which sit in a
    # different scale from the VGGT poses the models were trained on.
    scales_path: Optional[str] = None
    sample_sequence_prob: float = 0.0     # The probability of sampling frames in a sequence for video fine-tuning.
    sample_sequence_stride: int = 1       # Stride between target views when sampling a sequence for video fine-tuning.

class DatasetDL3DV(IterableDataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self._pass_idx = 0   # advances across epochs; see __iter__

        # Captured here, in the main process (dataloaders are built after the DDP process group
        # exists), and kept as plain ints so forked worker processes inherit correct values
        # without ever touching the process group themselves. Used to shard chunks over the full
        # (rank x worker) grid in __iter__.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.global_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.global_rank = 0
            self.world_size = 1

        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        self.load_depth_labels = cfg.load_depth_labels
        self.load_other_cameras = cfg.load_other_cameras
        self.sample_sequence_prob = cfg.sample_sequence_prob
        self.sample_sequence_stride = cfg.sample_sequence_stride

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            self.chunks.extend(root_chunks)

        # For DL3DV, we process up to the first 370 frames: VGGT/Video-Depth-Anything
        # camera and depth predictions are capped at 370 frames per scene due to memory
        # constraints during pre-processing, so any scene listed here must have its
        # in-memory images truncated to match, or the view sampler could draw an index
        # valid for images/original cameras but out of bounds for the (shorter) VGGT
        # depth/camera arrays.
        self.scales = None
        self.skipped_chunk_frames = {}
        skipped_frames_lines = []
        skipped_frames_path = cfg.labels_root / self.data_stage / "skipped_frames"
        if os.path.exists(skipped_frames_path):
            skipped_frames_files = list(skipped_frames_path.glob("skipped_frames_scene_*.text"))
            for file in skipped_frames_files:
                with open(file, 'r') as f:
                    skipped_frames_lines += [line.strip() for line in f.readlines()]
        
        if self.data_stage == 'train':
            pattern = re.compile(r"^(\d{6})_dl3dv_([a-f0-9]+)$")
        else:
            pattern = re.compile(r"^(\d{6})_([a-f0-9]+)$")

        for line in skipped_frames_lines:
            match = pattern.match(line)
            chunk_id, scene_id = match.groups()
            if chunk_id in self.skipped_chunk_frames:
                self.skipped_chunk_frames[chunk_id].append(scene_id)
            else:
                self.skipped_chunk_frames[chunk_id] = [scene_id]

        # Resolve each label root independently, preferring a nested depths/ or cameras/ when it is
        # actually there (see label_paths.resolve_label_dir). This used to hinge on
        # labels_root == cameras_root and load_depth_labels, which silently sent a camera-only
        # bundle to the flat path; and it is now the same rule dataset_re10k uses.
        extra_root = (resolve_label_dir(self.cfg.labels_root, self.data_stage, "depths")
                      if self.cfg.labels_root is not None else None)
        cameras_dir = (resolve_label_dir(self.cfg.cameras_root, self.data_stage, "cameras")
                       if self.cfg.cameras_root is not None else None)
        confs_root_sub = None
        if self.load_depth_labels and self.data_stage == "train" and self.cfg.labels_root is not None:
            candidate = self.cfg.labels_root / self.data_stage / "confs"
            confs_root_sub = candidate if os.path.exists(candidate) else None

        if self.load_depth_labels:
            # Collect extra labels (e.g., depth)
            suffix = ".zarr" if self.cfg.load_labels_from_zarr_dirs else ".npz"
            self.extra_chunks = sorted(
                [path for path in extra_root.iterdir() if path.suffix == suffix]
            )

            if confs_root_sub:
                self.confs_chunks = sorted(
                    [path for path in confs_root_sub.iterdir() if path.suffix == ".npz"]
                )
            else:
                self.confs_chunks = None

        if self.load_other_cameras:
            # Load cameras from another source. 
            # This is useful where loading cameras predicted with a different method rather than COLMAP.
            cameras_root = cameras_dir
            suffix = ".npz"
            self.camera_chunks = sorted(
                [path for path in cameras_root.iterdir() if path.suffix == suffix]
            )

        if cfg.scales_path is not None:
            with open(cfg.scales_path, 'r') as out:
                self.scales = json.load(out)
        elif cfg.labels_root is not None and self.load_depth_labels:
            # Check if a scales.json file exists in the same folder as the chunks.
            scales_path = cfg.labels_root / self.data_stage / "scales.json"
            if scales_path.exists():
                with open(scales_path, 'r') as out:
                    self.scales = json.load(out)
            else:
                self.scales = None

        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            chunk_id = os.path.basename(chunk_path).replace(".torch", "")
            self.chunks = [chunk_path] * len(self.chunks)
            if self.load_depth_labels:  
                if not self.cfg.load_labels_from_zarr_dirs:
                    suffix = "npz"
                else:
                    suffix = "zarr"
                
                chunk_extra_path = os.path.join(extra_root, f"{chunk_id}.{suffix}")

                self.extra_chunks = [chunk_extra_path] * len(self.chunks) 

            if self.load_other_cameras:
                camera_chunk_path = os.path.join(cameras_dir, f"{chunk_id}.npz")

                self.camera_chunks = [camera_chunk_path] * len(self.chunks)

        if self.stage == "test":
            # Fast testing.
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
            if self.load_depth_labels:
                self.extra_chunks = self.extra_chunks[:: cfg.test_chunk_interval]
            if self.load_other_cameras:
                self.camera_chunks = self.camera_chunks[:: cfg.test_chunk_interval]

        if self.stage == "val":
            self.chunks = self.chunks * int(1e6 // len(self.chunks))
            if self.load_depth_labels:
                self.extra_chunks = self.extra_chunks * int(1e6 // len(self.extra_chunks))
            if self.load_other_cameras:
                self.camera_chunks = self.camera_chunks * int(1e6 // len(self.camera_chunks))

        # print(self.chunks)
        # print(self.extra_chunks)
        # print(self.camera_chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def _shard_chunks(self, pass_idx: int) -> tuple[list, list, list | None, list]:
        """This (rank, worker)'s disjoint slice of the chunk lists for one pass over the data.

        Returns local lists and never mutates self.chunks/extra_chunks/confs_chunks/camera_chunks:
        with persistent_workers=True the dataset object survives across passes, so reassigning the
        attributes here would re-shard an already-sharded list and shrink the visible dataset by a
        factor of num_workers on every subsequent pass.
        """
        order = list(range(len(self.chunks)))

        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage == "train":
            # The seed depends only on pass_idx, so every rank and every worker draws the SAME
            # permutation. That is what makes the shard below a true partition (each chunk goes
            # to exactly one (rank, worker)) and gives every rank an identical chunk count --
            # per-worker RNG here instead would both duplicate/drop chunks and desynchronize the
            # ranks' stream lengths.
            generator = torch.Generator().manual_seed(SHARD_SHUFFLE_SEED + pass_idx)
            order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        elif self.stage == "val" and self.cfg.shuffle_val:
            # Legacy behavior on purpose: draw from the worker's global RNG, which is seeded
            # deterministically from data_loader.val.seed, so validation walks the same chunk order
            # as pre-refactor runs and stays comparable with previously logged results. The shared
            # seed above is only needed to keep the train shard a partition; val uses a single
            # worker, is not rank-sharded, and ValidationWrapper takes one scene, so none of that
            # applies here.
            order = [order[i] for i in torch.randperm(len(order)).tolist()]

        # Shard over the full (rank x worker) grid, so all world_size * num_workers loader
        # processes read disjoint chunks instead of duplicating each other's I/O.
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        # Rank-shard for training only. val/test keep the previous single-pool behavior so
        # evaluation semantics (which scenes each process sees) are unchanged by this refactor.
        world_size = self.world_size if self.stage == "train" else 1
        global_rank = self.global_rank if self.stage == "train" else 0
        num_shards = world_size * num_workers
        # Rank-minor shard ids (worker * world_size + rank, rather than rank * num_workers +
        # worker) so that when len(chunks) is not divisible by num_shards the leftover chunks
        # spread one-per-rank instead of piling all of them onto rank 0.
        shard_id = worker_id * world_size + global_rank
        order = [i for position, i in enumerate(order) if position % num_shards == shard_id]

        chunks = [self.chunks[i] for i in order]
        extra_chunks = [self.extra_chunks[i] for i in order] if self.load_depth_labels else []
        confs_chunks = (
            [self.confs_chunks[i] for i in order]
            if self.load_depth_labels and self.data_stage == "train" and self.confs_chunks is not None
            else None
        )
        camera_chunks = [self.camera_chunks[i] for i in order] if self.load_other_cameras else []
        return chunks, extra_chunks, confs_chunks, camera_chunks

    def __iter__(self):
        # Training is step-based here (max_steps, step-based val_check_interval, max_epochs=-1),
        # so it never needs an epoch boundary -- and under DDP an epoch boundary is actively
        # dangerous: __len__ promises len(index) * train_times_per_scene samples, but chunks hold
        # variable numbers of scenes and each worker rounds up its own final partial batch, so a
        # rank whose actual stream falls short of that promise raises StopIteration early. It then
        # enters Lightning's epoch-end collectives while the other ranks are still issuing
        # gradient all-reduces; the ranks end up in different collectives and the job deadlocks
        # until the NCCL watchdog kills it. Cycling forever means no rank can ever run dry, so
        # every rank reaches Lightning's max_batches together. Val/test still make one pass.
        # NOTE: _pass_idx lives on the instance, NOT in a local itertools.count(). The DataLoader
        # builds a fresh dataset iterator at every epoch boundary (it does this even with
        # persistent_workers=True), so a local counter would restart at 0 each epoch and, because
        # the permutation seed is SHARD_SHUFFLE_SEED + pass_idx, replay a byte-identical batch
        # sequence every epoch -- i.e. no reshuffling at all across epochs.
        while True:
            yield from self._iter_pass(self._pass_idx)
            if self.stage != "train":
                return
            self._pass_idx += 1

    def _iter_pass(self, pass_idx: int):
        chunks, extra_chunks, confs_chunks, camera_chunks = self._shard_chunks(pass_idx)

        for chunk_idx, chunk_path in enumerate(chunks):
            real_chunk_id = chunk_path.stem
            # Load the chunk.
            chunk = torch.load(chunk_path)

            cropped_scenes = self.skipped_chunk_frames[real_chunk_id] if real_chunk_id in self.skipped_chunk_frames else []

            if self.load_depth_labels:
                if not self.cfg.load_labels_from_zarr_dirs:
                    extra_chunk = np.load(extra_chunks[chunk_idx])
                else:
                    # Just save the root Zarr directory path for the chunk.
                    # We can random-access single scenes with load_scene_from_zarr method.
                    extra_chunk = extra_chunks[chunk_idx]

                if self.data_stage == 'train' and confs_chunks is not None:
                    confs_chunk = np.load(confs_chunks[chunk_idx])
                else:
                    confs_chunk = None

            if self.load_other_cameras:
                camera_chunk = np.load(camera_chunks[chunk_idx])

            # Access chunk and "crop" scenes if there are scenes to crop at 370 frames.
            for scene in chunk:
                if scene['key'] in cropped_scenes:
                    scene['images'] = scene['images'][:370]
                
            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"]
                        == self.cfg.overfit_to_scene]
                assert len(item) == 1
                if self.stage == "test":
                    chunk = item
                else:
                    chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                self.cfg.test_times_per_scene
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]
                scene = example["key"]
                
                if scene in SKIP_SCENES:
                    continue

                if self.load_other_cameras:
                    # NOTE: When loading intrinsics predicted by VGGT, note that these are not normalized!
                    cameras = torch.from_numpy(camera_chunk[scene])
                else:
                    cameras = example["cameras"]

                extrinsics, intrinsics = self.convert_poses(cameras)

                if self.load_depth_labels:
                    if self.cfg.load_labels_from_zarr_dirs:
                        scene_depth = load_scene_from_zarr(str(extra_chunk), scene)
                        scene_depth = torch.from_numpy(scene_depth)
                    else:
                        scene_depth = torch.from_numpy(extra_chunk[scene])
                    _, Hd, Wd = scene_depth.shape

                    if self.data_stage == 'train' and confs_chunk is not None:
                        scene_depth_conf = torch.from_numpy(confs_chunk[scene])
                    else:
                        scene_depth_conf = None

                context_depth_conf = None
                target_depth_conf = None

                try:
                    extra_kwargs = {}
                    if self.cfg.overfit_to_scene is not None and self.stage != "test":
                        extra_kwargs.update(
                            {
                                "max_num_views": (
                                    148
                                    if self.cfg.overfit_max_views is None
                                    else self.cfg.overfit_max_views
                                )
                            }
                        )
                    # We may also validate with sequences (camera trajectories) sometimes.
                    if self.stage in ('train','val'):
                        sample_sequence = np.random.choice(
                            [True, False], 
                            1, 
                            p=[self.sample_sequence_prob, 1.0 - self.sample_sequence_prob]
                        ).item()
                    else:
                        sample_sequence = False

                    out_data = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                        min_context_views=self.cfg.min_views,
                        max_context_views=self.cfg.max_views,
                        sample_sequence=sample_sequence,
                        sample_sequence_stride=self.sample_sequence_stride,
                        **extra_kwargs,
                    )
                    
                    if isinstance(out_data, tuple):
                        context_indices, target_indices = out_data[:2]
                        c_list = [
                            (
                                context_indices.sort()[0]
                                if self.cfg.sort_context_index
                                else context_indices
                            )
                        ]
                        t_list = [
                            (
                                target_indices.sort()[0]
                                if self.cfg.sort_target_index
                                else target_indices
                            )
                        ]
                    if isinstance(out_data, list):
                        c_list = [
                            (
                                a.context.sort()[0]
                                if self.cfg.sort_context_index
                                else a.context
                            )
                            for a in out_data
                        ]
                        t_list = [
                            (
                                a.target.sort()[0]
                                if self.cfg.sort_target_index
                                else a.target
                            )
                            for a in out_data
                        ]

                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                for context_indices, target_indices in zip(c_list, t_list):
                    # Load the images.
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]

                    try:
                        context_images = self.convert_images(context_images)

                        # The loaded (VGGT) intrinsics are PIXEL-valued at the resolution VGGT ran
                        # at, and the crop shim centre-crops in pixel space -- so images must be
                        # brought to that resolution first or the crop, and hence the focal length,
                        # comes out wrong. With depth labels that is the depth resolution; with poses
                        # only it is recovered from the principal point, which VGGT centres.
                        if self.load_depth_labels:
                            resize_to = (Hd, Wd)
                        elif self.load_other_cameras:
                            resize_to = (int(round(2 * intrinsics[0, 1, 2].item())),
                                         int(round(2 * intrinsics[0, 0, 2].item())))
                        else:
                            # The chunks' own poses carry NORMALIZED intrinsics, which are
                            # resolution-independent, so any target works -- but the native 270x480
                            # would fail the skip_bad_shape check below, so use the same resolution
                            # the rest of the pipeline is built around.
                            resize_to = LABEL_RESOLUTION

                        if resize_to is not None:
                            context_images = F.interpolate(context_images, size=resize_to,
                                                           mode="bilinear", align_corners=False)
                        if self.load_depth_labels:
                            context_depth = scene_depth[context_indices]

                            if scene_depth_conf is not None:
                                context_depth_conf = scene_depth_conf[context_indices]
                            else:
                                context_depth_conf = None

                            # print(f'context_depth.shape = {context_depth.shape}')
                    except OSError:
                        # some data might be corrupted
                        continue

                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]

                    try:
                        target_images = self.convert_images(target_images)
                        if resize_to is not None:
                            target_images = F.interpolate(target_images, size=resize_to,
                                                          mode="bilinear", align_corners=False)
                        if self.load_depth_labels:
                            target_depth = scene_depth[target_indices]

                            if scene_depth_conf is not None:
                                target_depth_conf = scene_depth_conf[target_indices]
                            else:
                                target_depth_conf = None
                    except OSError:
                        # Some data might be corrupted
                        continue
                        
                    # The chunks' own (COLMAP) intrinsics are NORMALIZED while the crop shim
                    # centre-crops in pixel space; VGGT's are already pixel-valued. Same conversion
                    # as dataset_re10k -- without it the principal point lands outside the image.
                    if not self.load_other_cameras:
                        h_img, w_img = context_images.shape[-2:]
                        intrinsics = intrinsics.clone()
                        intrinsics[:, 0, 0] *= w_img
                        intrinsics[:, 0, 2] *= w_img
                        intrinsics[:, 1, 1] *= h_img
                        intrinsics[:, 1, 2] *= h_img

                    if self.scales is not None:
                        scale = self.scales.get(scene, 1.0)
                        extrinsics[:, :3, 3] *= scale
                    else:
                        scale = 1.0

                    # Skip the example if the images don't have the right shape.
                    expected_shape = (3, *LABEL_RESOLUTION)

                    context_image_invalid = context_images.shape[1:] != expected_shape
                    target_image_invalid = target_images.shape[1:] != expected_shape

                    if self.cfg.skip_bad_shape and (
                        context_image_invalid or target_image_invalid
                    ):
                        print(
                            f"Skipped bad example {example['key']}. Context shape was "
                            f"{context_images.shape}, target shape was "
                            f"{target_images.shape}, and expected shape was {expected_shape}"
                        )
                        continue

                    # check the extrinsics
                    if any(torch.isnan(torch.det(extrinsics[context_indices][:, :3, :3]))):
                        # print('invalid extrinsics')
                        continue

                    if any(torch.isnan(torch.det(extrinsics[target_indices][:, :3, :3]))):
                        # print('invalid extrinsics')
                        continue

                    # check the extrinsics: translation could be very large
                    # https://github.com/DL3DV-10K/Dataset/issues/34
                    if (extrinsics[context_indices][:, :3, 3] > 1e3).any():
                        # print('extremely large camera translation')
                        continue

                    if (extrinsics[target_indices][:, :3, 3] > 1e3).any():
                        # print('extremely large camera translation')
                        continue

                    if not torch.allclose(torch.det(extrinsics[context_indices][:, :3, :3]), torch.det(extrinsics[context_indices][:, :3, :3]).new_tensor(1)):
                        # print('invalid extrinsics')
                        continue
                    if not torch.allclose(torch.det(extrinsics[target_indices][:, :3, :3]), torch.det(extrinsics[target_indices][:, :3, :3]).new_tensor(1)):
                        # print('invalid extrinsics')
                        continue
                    
                    # print(scene, context_indices, target_indices)
                    nf_scale = 1.0
                    example_out = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            # Omitted, not None: the default collate rejects None, so an absent key is
                            # the only representation of "no depth labels" that survives the DataLoader.
                            **({"depth": context_depth * scale} if self.load_depth_labels else {}),
                            "near": self.get_bound("near", len(context_indices))
                            / nf_scale,
                            "far": self.get_bound("far", len(context_indices))
                            / nf_scale,
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            # Omitted, not None: the default collate rejects None, so an absent key is
                            # the only representation of "no depth labels" that survives the DataLoader.
                            **({"depth": target_depth * scale} if self.load_depth_labels else {}),
                            "near": self.get_bound("near", len(target_indices))
                            / nf_scale,
                            "far": self.get_bound("far", len(target_indices))
                            / nf_scale,
                            "index": target_indices,
                        },
                        "scene": scene,
                    }

                    if self.stage == "train" and context_depth_conf is not None:
                        example_out["context"]["mask"] = context_depth_conf

                    if self.stage == "train" and target_depth_conf is not None:
                        example_out["target"]["mask"] = target_depth_conf

                    if self.stage == "train" and self.cfg.augment:
                        example_out = apply_augmentation_shim(example_out)
                    if self.cfg.image_shape == list(context_images.shape[2:]):
                        yield example_out
                    else:
                        yield apply_crop_shim(example_out, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                if not (root / data_stage).is_dir():
                    continue

                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v)
                         for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if self.stage in ['train', 'test']:
            return (
                min(
                    len(self.index.keys()) * self.cfg.test_times_per_scene,
                    self.cfg.test_len,
                )
                if self.stage == "test" and self.cfg.test_len > 0
                else len(self.index.keys()) * self.cfg.train_times_per_scene
            )
        else:
            # Set a very large value here to ensure the validation keep going
            # and do not exhaust; it will be wrap to length 1 anyway.
            return int(1e10)
