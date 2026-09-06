from dataclasses import dataclass

from jaxtyping import Float
from typing import Optional
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..model.diffusion_adapter import DiffusionAdapter
from ..misc.nn_module_tools import convert_to_buffer
from einops import rearrange
from lpips import LPIPS
import torch
from ..misc.types import DiffuserOutput

@dataclass
class LossDiffusionCfg:
    c_weight: float
    d_weight: float
    apply_after_step: Optional[int] = 0
    disable_after_step: Optional[int] = int(1e7)
    use_confidence_mask: bool = False  # Re-weight the depth loss by batch["target"]["mask"] when available (DL3DV only).

@dataclass
class LossDiffusionCfgWrapper:
    diffusion: LossDiffusionCfg

class LossDiffusion(Loss[LossDiffusionCfg, LossDiffusionCfgWrapper]):
    def forward(
        self,
        diff_output: DiffuserOutput,
        batch: BatchedExample,
        global_step: int,
    ) -> Float[Tensor, ""]:

        color_prediction, color_gt = diff_output.pred_color, diff_output.gt_color
        depth_prediction, depth_gt = diff_output.pred_depth, diff_output.gt_depth

        depth_mask = diff_output.valid_mask_depth

        if global_step < self.cfg.apply_after_step or global_step > self.cfg.disable_after_step:
            return torch.tensor(0.0, dtype=torch.float32, device=color_gt.device)

        color_delta = color_prediction - color_gt

        if depth_prediction is not None: # ablation exp. with disabled depth modeling. 
            
            if self.cfg.use_confidence_mask and "mask" in batch["target"] and batch["target"]["mask"] is not None:
                conf_mask = batch["target"]["mask"]
                hl, wl = depth_prediction.shape[-2:]
                hc, wc = conf_mask.shape[-2:]
                down_factor = hc // hl 
                assert down_factor == wc // wl
                conf_mask = rearrange(conf_mask, "b v h w -> (b v) h w")
                conf_mask = torch.nn.functional.interpolate(
                    conf_mask.unsqueeze(1).float(),   # [B,1,H,W]
                    size=(hl, wl),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)   
                conf_mask = rearrange(conf_mask, "(b v) h w -> b v h w", b=depth_prediction.shape[0])
                # confidence_mask should already be resized to the same spatial size
                conf_mask = conf_mask.clamp_min(0.0)
                conf_mask = conf_mask.unsqueeze(2).float().expand_as(depth_prediction)
                conf_mask = conf_mask / conf_mask.mean().clamp_min(1e-8)
            else:
                conf_mask = None

            if depth_mask is not None:
                depth_mask = depth_mask.float()
                depth_mask = depth_mask.expand_as(depth_prediction)
                if conf_mask is not None:
                    depth_mask = depth_mask * conf_mask
                depth_delta = (depth_prediction - depth_gt) * depth_mask
                depth_loss = (depth_delta**2).sum() / depth_mask.sum().clamp_min(1.0)
            else:
                depth_delta = depth_prediction - depth_gt
                depth_loss = (depth_delta**2).mean()

            loss = self.cfg.c_weight * (color_delta**2).mean() + self.cfg.d_weight * depth_loss
        else:
            loss = self.cfg.c_weight * (color_delta**2).mean()

        return loss