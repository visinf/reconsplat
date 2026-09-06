from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from jaxtyping import Float
import torch
from torch import Tensor

from .loss import Loss
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..model.first_stage.adapter import FirstStageAdapter
from einops import rearrange, repeat
from ..model.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ..misc.depth_io import normalize_depth
from functools import partial
import torchmetrics.functional as F

def ssim_depth(d1, d2, mask=None):
    if mask is not None:
        d1 = d1 * mask
        d2 = d2 * mask
    # expects shape [B, C, H, W]
    if d1.ndim == 3:  # [B, H, W]
        d1, d2 = d1.unsqueeze(1), d2.unsqueeze(1)
    return F.structural_similarity_index_measure(
        d1, d2, data_range=1.0, kernel_size=5
    )

@dataclass
class LossDepthVAESSIMCfg:
    weight: float = 1.0
    apply_after_step: int = 0
    disable_after_step: int = int(1e7)

@dataclass
class LossDepthVAESSIMCfgWrapper:
    vae_depth_ssim: LossDepthVAESSIMCfg

class LossDepthVAESSIM(Loss[LossDepthVAESSIMCfg, LossDepthVAESSIMCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        autoencoder: FirstStageAdapter,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if global_step < self.cfg.apply_after_step or global_step > self.cfg.disable_after_step:
            return torch.tensor(0, dtype=torch.float32, device=batch["target"]["image"].device)
        
        gt = rearrange(batch['target']['depth'], 'b v h w -> (b v) h w')

        # Normalize depth to range [-1,1] based on near and far values
        near = batch["target"]["near"]
        far = batch["target"]["far"]
        near = repeat(near, "b v -> (b v)")
        far = repeat(far, "b v -> (b v)")

        # Compute valid mask based on fixed near and far ranges, to avoid outliers in GT.
        valid_mask = (gt > near[:, None, None]) & (gt < far[:, None, None])

        gt = torch.maximum(torch.minimum(gt, far[:, None, None]), near[:, None, None])
        # Normalize depth to range [0, 1]
        gt = (gt - near[:, None, None]) / (far[:, None, None] - near[:, None, None])
        gt = repeat(gt, "b h w -> b c h w", c=3)

        # Get predicted depth, in range [near, far], so normalize.
        pred = (prediction.depth_decoded - near[:, None, None, None]) / (far[:, None, None, None] - near[:, None, None, None])

        return self.cfg.weight * (1 - ssim_depth(gt, pred, valid_mask[:, None, ...]))