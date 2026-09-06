from dataclasses import dataclass

from jaxtyping import Float
from typing import Optional
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..model.first_stage.adapter import FirstStageAdapter
from einops import rearrange, repeat

import torch

@dataclass
class LossDepthVAECharbonnierCfg:
    weight: float
    grad_weight: float = 0.1
    apply_after_step: Optional[int] = 0
    disable_after_step: Optional[int] = int(1e7)

@dataclass
class LossDepthVAECharbonnierCfgWrapper:
    vae_depth_charb: LossDepthVAECharbonnierCfg
    
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def grad_xy(t):
    dx = t[..., :, 1:] - t[..., :, :-1]
    dy = t[..., 1:, :] - t[..., :-1, :]
    return dx, dy

# NOTE: revert back to compute loss with 3-channel depth.

class LossDepthVAECharbonnier(Loss[LossDepthVAECharbonnierCfg, LossDepthVAECharbonnierCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        autoencoder: FirstStageAdapter,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        device = batch["target"]["image"].device
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step or global_step > self.cfg.disable_after_step:
            return torch.tensor(0, dtype=torch.float32, device=device)

        # Get ground-truth depth.
        gt = rearrange(batch["target"]["depth"], "b v h w -> (b v) h w")
        near = rearrange(batch["target"]["near"], "b v -> (b v)")
        far = rearrange(batch["target"]["far"], "b v -> (b v)")
        
        pred = prediction.depth_decoded
        log_pred = torch.log(pred)

        # Compute valid mask based on fixed near and far ranges, to avoid outliers in GT.
        valid_mask = (gt > near[:, None, None]) & (gt < far[:, None, None])
        
        gt = torch.maximum(torch.minimum(gt, far[:, None, None]), near[:, None, None])
        gt = repeat(gt, "b h w -> b c h w", c=3)
        log_gt = torch.log(gt)        

        # Compute Charbonnier loss between decoded (log-)depth and (log-)gt.
        delta = (log_gt - log_pred) * valid_mask[:, None, ...]
        charb = charbonnier(delta).sum() / (valid_mask.sum() + 1e-8) 

        # Now compute gradient loss with gt.
        dx_p, dy_p = grad_xy(log_pred)
        dx_g, dy_g = grad_xy(log_gt)

        # Compute valid masks for gradients.
        vm_x = valid_mask[..., :, 1:] * valid_mask[..., :, :-1]
        vm_y = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]
        vm_x = vm_x[:, None, ...]
        vm_y = vm_y[:, None, ...]
        
        grad = (
            charbonnier((dx_p - dx_g) * vm_x).sum() / (vm_x.sum() + 1e-8)
            + charbonnier((dy_p - dy_g) * vm_y).sum() / (vm_y.sum() + 1e-8)
        ) 

        return self.cfg.weight * charb + self.cfg.grad_weight * grad 
