from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
from typing import Optional

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]
    color_feature_harmonics: Float[Tensor, "*batch channels _"] | None
    color_features: Float[Tensor, "*batch channels"] | None
    geometry_features: Float[Tensor, "*batch channels"] | None

@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    feature_sh_degree: Optional[int] = 0
    n_feature_channels: Optional[int] = 0
    use_feature_sh: bool = False
    variational_features: bool = True

class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.n_feature_channels = cfg.n_feature_channels * 2 if cfg.variational_features else cfg.n_feature_channels
        if self.n_feature_channels:
            assert (
                self.cfg.feature_sh_degree > 0
            ), "set feature_sh_degree > 0 to enable feature prediction"
            self.register_buffer(
                "feature_sh_mask",
                torch.ones((self.d_feature_sh,), dtype=torch.float32),
                persistent=False,
            )
            for degree in range(1, self.cfg.feature_sh_degree + 1):
                self.feature_sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        if self.n_feature_channels > 0:
            scales, rotations, sh, color_feature, geometry_feature = raw_gaussians.split(
                (
                    3,
                    4,
                    3 * self.d_sh,
                    self.n_feature_channels * self.d_feature_sh if self.cfg.use_feature_sh else self.n_feature_channels,
                    self.n_feature_channels
                ),
                dim=-1,
            )
            color_feature_sh = color_feature if self.cfg.use_feature_sh else None
        else:
            scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
            color_feature, geometry_feature = None, None
            color_feature_sh = None

        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        if color_feature_sh is not None:
            color_feature_sh = rearrange(
                color_feature_sh, "... (c d_sh) -> ... c d_sh", c=self.n_feature_channels
            )
            color_feature_sh = (
                color_feature_sh.broadcast_to(
                    (*opacities.shape, self.n_feature_channels, self.d_feature_sh)
                )
                * self.feature_sh_mask
            )

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        if self.cfg.variational_features:
            if color_feature is None:
                raise ValueError("color_feature cannot be None when using variational features")
            color_mean, color_std = torch.chunk(color_feature, 2, dim=-1) 
            # Use softplus activation to ensure std is positive
            color_std = torch.nn.functional.softplus(color_std) + 1e-3
            color_feature = torch.cat([color_mean, color_std], dim=-1)
            
            if geometry_feature is not None:
                geom_mean, geom_std = torch.chunk(geometry_feature, 2, dim=-1)
                geom_std = torch.nn.functional.softplus(geom_std) + 1e-3
                geometry_feature = torch.cat([geom_mean, geom_std], dim=-1)

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            color_feature_harmonics=(
                None
                if color_feature_sh is None
                else rotate_sh(color_feature_sh, c2w_rotations[..., None, :, :])
            ),
            color_features=color_feature,
            geometry_features=geometry_feature,
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_feature_sh(self) -> int:
        return (self.cfg.feature_sh_degree + 1) ** 2
    
    @property
    def d_in(self) -> int:
        if self.cfg.use_feature_sh:
            return 7 + 3 * self.d_sh + self.n_feature_channels * (self.d_feature_sh + 1) 
        return 7 + 3 * self.d_sh + self.n_feature_channels * 2
