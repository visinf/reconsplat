'''
    Custom loader for RealEstate10K dataset. Same structure as the loader from DepthSplat, 
    with support for loading depth labels and other cameras from a different source (e.g., VGGT).
'''

import itertools
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import os
import numpy as np
import torch
import torchvision.transforms as tf
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon, SHARD_SHUFFLE_SEED
from .shims.augmentation_shim import apply_augmentation_shim
from .label_paths import resolve_label_dir
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.depth_io import load_scene_from_zarr

@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    labels_root: Path | None
    load_depth_labels: bool
    baseline_epsilon: float
    max_fov: float
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    shuffle_val: bool = True
    sort_target_index: bool = False
    load_labels_from_zarr_dirs: bool = True
    load_other_cameras: bool = True         # Whether we should load other cameras that those saved in chunks.
    # Per-scene multiplier on the camera translations, as {scene: scale}. Needed when evaluating on
    # the chunks' own (COLMAP) poses, which sit in a different scale from the VGGT poses the models
    # were trained on -- about 4x larger on RE10K, and varying by scene, so the fixed near/far would
    # otherwise clip most of the scene away. Independent of load_depth_labels.
    scales_path: Optional[str] = None
    cameras_root: Path | None = None        
    chunk_idx_path: Optional[str] = None
    sample_sequence_prob: float = 0.0       # The probability of sampling frames in a sequence for video fine-tuning.
    sample_sequence_stride: int = 1         # Stride between target views when sampling a sequence for video fine-tuning.

class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
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
        self.scales = None
        self.sample_sequence_prob = cfg.sample_sequence_prob
        self.sample_sequence_stride = cfg.sample_sequence_stride

        # Collect chunks.   
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)

        if self.load_depth_labels:
            # Collect extra labels (e.g., depth). Same resolution rule as dataset_dl3dv.
            extra_root = resolve_label_dir(cfg.labels_root, self.data_stage, "depths")
            suffix = ".zarr" if self.cfg.load_labels_from_zarr_dirs else ".npz"
            self.extra_chunks = sorted(
                [path for path in extra_root.iterdir() if path.suffix == suffix]
            )

        if self.load_other_cameras:
            # Load cameras from another source. Same resolution rule as dataset_dl3dv, so a
            # processed tree (<root>/<split>/cameras/) and a camera-only bundle
            # (<root>/<split>/) both work here too.
            cameras_root = resolve_label_dir(cfg.cameras_root, self.data_stage, "cameras")
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
                # Setup the Zarr root folder for the chunk, via the same resolver as above so the
                # overfit path cannot disagree with the listing path about the layout.
                chunk_extra_path = os.path.join(extra_root, f"{chunk_id}.{suffix}")
                self.extra_chunks = [chunk_extra_path] * len(self.chunks) 

            if self.load_other_cameras:
                camera_chunk_path = os.path.join(cameras_root, f"{chunk_id}.npz")
                self.camera_chunks = [camera_chunk_path] * len(self.chunks)

        if self.stage == "test":
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
            if self.load_depth_labels:
                self.extra_chunks = self.extra_chunks[:: cfg.test_chunk_interval]
            if self.load_other_cameras:
                self.camera_chunks = self.camera_chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst1: list) -> list:
        indices = torch.randperm(len(lst1))
        return [lst1[x] for x in indices]

    def _shard_chunks(self, pass_idx: int) -> tuple[list, list, list]:
        """This (rank, worker)'s disjoint slice of the chunk lists for one pass over the data.

        Returns local lists and never mutates self.chunks/extra_chunks/camera_chunks: with
        persistent_workers=True the dataset object survives across passes, so reassigning the
        attributes here would re-shard an already-sharded list and shrink the visible dataset by
        a factor of num_workers on every subsequent pass.
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
        camera_chunks = [self.camera_chunks[i] for i in order] if self.load_other_cameras else []
        return chunks, extra_chunks, camera_chunks

    def __iter__(self):
        # Training is step-based here (max_steps, step-based val_check_interval, max_epochs=-1),
        # so it never needs an epoch boundary -- and under DDP an epoch boundary is actively
        # dangerous: chunks hold variable numbers of scenes and each worker rounds up its own
        # final partial batch, so per-rank stream lengths differ slightly. The first rank to
        # exhaust its stream then enters Lightning's epoch-end collectives while the others are
        # still issuing gradient all-reduces; the ranks end up in different collectives and the
        # job deadlocks until the NCCL watchdog kills it. Cycling forever keeps every rank in
        # lockstep on training steps indefinitely. Val/test still make exactly one pass.
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
        chunks, extra_chunks, camera_chunks = self._shard_chunks(pass_idx)

        if self.cfg.overfit_to_scene:
            chunk_path = chunks[0]
            # Cache block and scene labels when overfitting.
            cache_chunk = torch.load(chunk_path)
            cache_extra_chunk = cache_scene_depth = None
            # extra_chunks is empty when load_depth_labels is off, so this whole block has to be
            # skipped for a poses-only run rather than indexing into it.
            if self.load_depth_labels:
                if not self.cfg.load_labels_from_zarr_dirs:
                    cache_extra_chunk = np.load(extra_chunks[0])
                else:
                    # Just save the root Zarr directory path for the chunk.
                    # We can random-access single scenes with load_scene_from_zarr.
                    cache_extra_chunk = extra_chunks[0]
                cache_scene_depth = load_scene_from_zarr(str(cache_extra_chunk), self.cfg.overfit_to_scene)
                cache_scene_depth = torch.from_numpy(cache_scene_depth)

            if self.load_other_cameras:
                cache_camera_chunk = np.load(camera_chunks[0])

        for chunk_idx, chunk_path in enumerate(chunks):

            if self.cfg.overfit_to_scene:
                chunk = cache_chunk
                extra_chunk = cache_extra_chunk
                camera_chunk = cache_camera_chunk
                scene_depth = cache_scene_depth
                # Hd/Wd are only read inside load_depth_labels branches below.
                if scene_depth is not None:
                    _, Hd, Wd = scene_depth.shape
            else:
                chunk = torch.load(chunk_path)
    
                if self.load_depth_labels:
                    if not self.cfg.load_labels_from_zarr_dirs:
                        extra_chunk = np.load(extra_chunks[chunk_idx])
                    else:
                        # Just save the root Zarr directory path for the chunk.
                        # We can random-access single scenes with load_scene_from_zarr.
                        extra_chunk = extra_chunks[chunk_idx]

                if self.load_other_cameras:
                    camera_chunk = np.load(camera_chunks[chunk_idx])

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)
    
            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                # No need to permute 
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = (
                self.cfg.test_times_per_scene
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                if times_per_scene > 1:
                    # Disambiguate repeated visits to the same scene within an epoch.
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                else:
                    scene = example["key"]

                if self.load_other_cameras:
                    # NOTE: When loading intrinsics predicted by VGGT, note that these are not normalized!
                    cameras = torch.from_numpy(camera_chunk[scene])
                else:
                    cameras = example["cameras"]

                extrinsics, intrinsics = self.convert_poses(cameras)

                if self.cfg.overfit_to_scene is None:
                    if self.load_depth_labels:
                        if self.cfg.load_labels_from_zarr_dirs:
                            scene_depth = load_scene_from_zarr(str(extra_chunk), scene)
                            scene_depth = torch.from_numpy(scene_depth)
                        else:
                            scene_depth = torch.from_numpy(extra_chunk[scene])
                        _, Hd, Wd = scene_depth.shape
                        
                try:
                    # We may also validate with sequences sometimes.
                    if self.stage in ('train','val'):
                        sample_sequence = np.random.choice(
                            [True, False], 
                            1, 
                            p=[self.sample_sequence_prob, 1.0 - self.sample_sequence_prob]
                        ).item()
                    else:
                        sample_sequence = False

                    sampled_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                        sample_sequence=sample_sequence,
                        sample_sequence_stride=self.sample_sequence_stride
                    )

                    if len(sampled_indices) == 2:
                        context_indices, target_indices = sampled_indices
                        target_quantify_indices = None
                    if len(sampled_indices) == 3:
                        context_indices, target_indices, target_quantify_indices = (
                            sampled_indices
                        )

                    if not sample_sequence and self.cfg.sort_target_index:
                        target_indices = target_indices.sort()[0]

                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue
                
                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)

                # The loaded (VGGT) intrinsics are PIXEL-valued at the resolution VGGT ran at, and
                # the crop shim does its centre-crop in pixel space -- so the images have to be
                # brought to that resolution first, or the crop, and hence the focal length, comes
                # out wrong. With depth labels that resolution is the depth resolution. With poses
                # only it is recovered from the principal point, which VGGT puts at the image centre.
                if self.load_depth_labels:
                    resize_to = (Hd, Wd)
                elif self.load_other_cameras:
                    resize_to = (int(round(2 * intrinsics[0, 1, 2].item())),
                                 int(round(2 * intrinsics[0, 0, 2].item())))
                else:
                    resize_to = None

                if resize_to is not None:
                    context_images = F.interpolate(context_images, size=resize_to, mode="bilinear",
                                                   align_corners=False)
                if self.load_depth_labels:
                    context_depth = scene_depth[context_indices]
                    
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)
                # print(f'target_images.shape = {target_images.shape}')

                if resize_to is not None:
                    target_images = F.interpolate(target_images, size=resize_to, mode="bilinear",
                                                  align_corners=False)
                if self.load_depth_labels:
                    target_depth = scene_depth[target_indices]

                # The chunk's own (COLMAP) intrinsics are NORMALIZED, but the crop shim applied on
                # the way out centre-crops in PIXEL space -- handing it normalized values puts the
                # principal point outside the image (measured cx = -0.39) and the focal length off by
                # ~700x. VGGT intrinsics are already pixel-valued at the resolution VGGT ran at, so
                # only the chunk source needs converting. Done after the get_fov check above, which
                # is the one place that does want normalized values.
                if not self.load_other_cameras:
                    h_img, w_img = context_images.shape[-2:]
                    intrinsics = intrinsics.clone()
                    intrinsics[:, 0, 0] *= w_img
                    intrinsics[:, 0, 2] *= w_img
                    intrinsics[:, 1, 1] *= h_img
                    intrinsics[:, 1, 2] *= h_img

                if self.scales is not None:
                    scale = self.scales.get(scene, 1.0)
                    # Scale extrinsics translation accordingly.
                    extrinsics[:, :3, 3] *= scale
                else:
                    scale = 1.0

                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        # Omitted, not None: the default collate rejects None, so an absent key is
                        # the only representation of "no depth labels" that survives the DataLoader.
                        **({"depth": context_depth * scale} if self.load_depth_labels else {}),
                        "near": self.get_bound("near", len(context_indices)),
                        "far": self.get_bound("far", len(context_indices)),
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        # Omitted, not None: the default collate rejects None, so an absent key is
                        # the only representation of "no depth labels" that survives the DataLoader.
                        **({"depth": target_depth * scale} if self.load_depth_labels else {}),
                        "near": self.get_bound("near", len(target_indices)),
                        "far": self.get_bound("far", len(target_indices)),
                        "index": target_indices,
                    },
                    "scene": scene,
                }
                if target_quantify_indices is not None:
                    example['target'].update({"index_quantify": target_quantify_indices})
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)

                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

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
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
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
                # Load the root's index.
                if self.cfg.chunk_idx_path is not None:
                    # print("use chunk idx", self.cfg.chunk_idx_path)
                    with open(self.cfg.chunk_idx_path, "r") as f:
                        index = json.load(f)
                else:
                    with (root / data_stage / "index.json").open("r") as f:
                        index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

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
            # set a very large value here to ensure the validation keep going
            # and do not exhaust; it will be wrap to length 1 anyway.
            return int(1e10)

