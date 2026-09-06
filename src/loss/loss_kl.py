from dataclasses import dataclass
from typing import Literal, Tuple

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
class LossFeatureKlCfg:
    color_weight: float = 1.0e-6    # Same as in CompVis/latent-diffusion first_stage_models configs.
    depth_weight: float = 1.0e-6 
    apply_after_step: int = 0
    disable_after_step: int = int(1e7)
    
@dataclass
class LossFeatureKlCfgWrapper:
    kl_feature: LossFeatureKlCfg

class LossFeatureKl(Loss[LossFeatureKlCfg, LossFeatureKlCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        autoencoder: FirstStageAdapter,
        global_step: int,
    ) -> Tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        
        if global_step < self.cfg.apply_after_step or global_step > self.cfg.disable_after_step:
            return (torch.tensor(0, dtype=torch.float32, device=batch["target"]["image"].device), \
                torch.tensor(0, dtype=torch.float32, device=batch["target"]["image"].device))
        
        color_kl = self.unweighted_loss(prediction.color_posterior)
        depth_kl = self.unweighted_loss(prediction.depth_posterior)

        return (self.cfg.color_weight * color_kl, self.cfg.depth_weight * depth_kl)
    
    def unweighted_loss(
        self,
        prediction: DiagonalGaussianDistribution,
        gt: DiagonalGaussianDistribution | None = None
    ) -> Float[Tensor, ""]:
        return prediction.kl().mean()

