from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput
from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from .cuda_splatting import RasterizerOutput

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]
    variational_features: bool = True

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        if gaussians.color_feature_harmonics is not None:
            color_feature_sh = (
                repeat(gaussians.color_feature_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v)
                if gaussians.color_feature_harmonics is not None
                else None
            )
        else:
            color_feature_sh = None

        render_output: RasterizerOutput = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            gaussian_color_feature_sh_coefficients=color_feature_sh,
            gaussian_color_features=repeat(gaussians.color_features, "b g c -> (b v) g c", v=v) if gaussians.color_features is not None else None,
            gaussian_geometry_features=repeat(gaussians.geometry_features, "b g c -> (b v) g c", v=v) if gaussians.geometry_features is not None else None
        )
        color = rearrange(render_output.color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(render_output.depth.squeeze(1), "(b v) h w -> b v h w", b=b, v=v)
        mask = rearrange(render_output.mask.squeeze(1), "(b v) h w -> b v h w", b=b, v=v)
        feature = render_output.feature

        if self.variational_features:
            mean, std = feature.chunk(2, dim=1) 
            color_mean, geom_mean = mean.chunk(2, dim=1)
            color_std, geom_std = std.chunk(2, dim=1)
            
            color_posterior = DiagonalGaussianDistribution(mean=color_mean, logvar=2*torch.log(color_std))
            depth_posterior = DiagonalGaussianDistribution(mean=geom_mean, logvar=2*torch.log(geom_std))
        else:
            mean = feature
            color_mean, geom_mean = mean.chunk(2, dim=1)
            color_posterior = DiagonalGaussianDistribution(mean=color_mean, logvar=2*torch.log(1e-8))
            depth_posterior = DiagonalGaussianDistribution(mean=geom_mean, logvar=2*torch.log(1e-8))

        return DecoderOutput(
            color=color, 
            depth=depth,
            mask=mask,
            color_posterior=color_posterior,
            depth_posterior=depth_posterior,
            color_decoded=None, # decoded later from sampled color
            depth_decoded=None, # decoded later from sampled depth
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
