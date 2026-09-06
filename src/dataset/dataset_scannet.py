import json
import os
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.camera_utils import denormalize_K 

@dataclass
class DatasetScanNetCfg(DatasetCfgCommon):
    name: Literal["scannet"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    sort_target_index: bool = False
    chunk_idx_path: Optional[str] = None
    use_only_indexed_scenes: Optional[bool] = False

class DatasetScanNet(Dataset):
    cfg: DatasetScanNetCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetScanNetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect scenes.
        test_scenes = self.read_text(cfg.roots[0] / "nvs_test_iphone.txt")
        # Filter out null scenes.
        asset_scenes = self.read_json(self.view_sampler.cfg.index_path)
        filtered_asset_scenes = [scene for scene in asset_scenes if asset_scenes[scene] is not None]

        # Sanity-check that asset scenes are actually part of the iPhone NVS test set.
        unexpected_scenes = sorted(set(filtered_asset_scenes) - set(test_scenes))
        assert not unexpected_scenes, (
            f"Scenes in {self.view_sampler.cfg.index_path} are not part of "
            f"{cfg.roots[0] / 'nvs_test_iphone.txt'}: {unexpected_scenes}"
        )

        self._scenes = filtered_asset_scenes
        print("scenes: \n", self._scenes)

    def read_text(self, path):
        with open(path, "r") as f:
            scenes = f.read().split("\n")[:-1]
        return scenes

    def read_json(self, path):
        with open(path, "r") as f:
            poses = json.load(f)
        return poses

    def __getitem__(self, idx):
        # Pick a scene.
        self.scene = self._scenes[idx]

        # Load cameras. 
        poses = self.read_json(
            self.cfg.roots[0] / self.scene / "iphone/transforms.json"
        )
        extrinsics, intrinsics = self.convert_poses(poses)

        try:
            sampled_indices = self.view_sampler.sample(
                self.scene,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            # Skip because the example doesn't have enough frames.
            return self.__getitem__((idx + 1) % len(self._scenes))

        if len(sampled_indices) == 2:
            context_indices, target_indices = sampled_indices
            target_quantify_indices = None
        if len(sampled_indices) == 3:
            context_indices, target_indices, target_quantify_indices = sampled_indices

        # Sort the indices to ensure that the context and target images are in the correct order.
        context_indices = context_indices.sort()[0]
        target_indices = target_indices.sort()[0]

        context_images = [
            poses["frames"][index.item()]["file_path"] for index in context_indices
        ]
        context_images, context_depth = self.convert_images(context_images)
        target_images = [
            poses["frames"][index.item()]["file_path"] for index in target_indices
        ]
        target_images, target_depth = self.convert_images(target_images)

        H, W = context_images.shape[-2:]
        intrinsics = denormalize_K(intrinsics, W, H)

        context_extrinsics = extrinsics[context_indices]
        if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
            # Resize the world to make the baseline 1.
            a, b = context_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_epsilon:
                print(
                    f"Skipped {self.scene} because of insufficient baseline "
                    f"{scale:.6f}"
                )
                return self.__getitem__((idx + 1) % len(self._scenes))
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "depth": context_depth,
                "near": self.get_bound("near", len(context_indices)) / nf_scale,
                "far": self.get_bound("far", len(context_indices)) / nf_scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "depth": target_depth,
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
            },
            "scene": self.scene,
        }

        return apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses,
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b = len(poses["frames"])

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses["fl_x"], poses["fl_y"], poses["cx"], poses["cy"]
        w, h = poses["w"], poses["h"]
        # NOTE: omit distortion for now
        intrinsics[:, 0, 0] = fx / w
        intrinsics[:, 1, 1] = fy / h
        intrinsics[:, 0, 2] = cx / w
        intrinsics[:, 1, 2] = cy / h

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        c2w = torch.stack(
            [
                torch.tensor(f["transform_matrix"])
                @ torch.diag(torch.tensor([1, -1, -1, 1])).to(dtype=torch.float32)
                for f in poses["frames"]
            ],
            dim=0,
        )  # (total_frames, 4, 4)
        return c2w, intrinsics

    def convert_images(
        self,
        names: list[str],
    ) -> tuple[
        Float[Tensor, "batch 3 height width"],  # images
        Float[Tensor, "batch height width"],    # depths
    ]:
        torch_images = []
        torch_depth = []
        for name in names:
            rgb_name = Path(f"{self.cfg.roots[0]}/{self.scene}/iphone/rgb/{name}")
            image = Image.open(rgb_name)
            torch_images.append(self.to_tensor(image))
            depth_name = Path(
                f"{self.cfg.roots[0]}/{self.scene}/iphone/depth/{Path(name).stem}.png"
            )
            depth = torch.from_numpy(np.array(Image.open(depth_name), dtype=np.float32))
            # Upscale depth to image size.
            depth = (
                F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    scale_factor=7.5,
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            torch_depth.append(depth)
        return torch.stack(torch_images), torch.stack(torch_depth)

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

    def __len__(self) -> int:
        return len(self._scenes)
