from dataclasses import dataclass
from typing import Literal

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

@dataclass
class LossColorVAEL1Cfg:
    weight: float = 1.0
    apply_after_step: int = 0
    disable_after_step: int = int(1e7)

@dataclass
class LossColorVAEL1CfgWrapper:
    l1: LossColorVAEL1Cfg

class LossColorVAEL1(Loss[LossColorVAEL1Cfg, LossColorVAEL1CfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        autoencoder: FirstStageAdapter,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # turn off before apply_after_step
        if global_step < self.cfg.apply_after_step or global_step > self.cfg.disable_after_step:
            return torch.tensor(0, dtype=torch.float32, device=batch["target"]["image"].device)
        
        # Get ground-truth appearance.
        gt = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")
        
        return self.cfg.weight * self.unweighted_loss(prediction.color_decoded, gt)
    
    def unweighted_loss(
        self,
        prediction: Tensor,
        gt: Tensor 
    ) -> Float[Tensor, ""]:
        delta = prediction - gt 
        return delta.abs().mean()
