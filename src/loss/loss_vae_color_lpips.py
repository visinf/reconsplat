from dataclasses import dataclass

from jaxtyping import Float
from typing import Optional
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..model.first_stage.adapter import FirstStageAdapter
from ..misc.nn_module_tools import convert_to_buffer
from einops import rearrange
from lpips import LPIPS

import torch

@dataclass
class LossColorVAELpipsCfg:
    weight: float
    apply_after_step: Optional[int] = 0
    disable_after_step: Optional[int] = int(1e7)

@dataclass
class LossColorVAELpipsCfgWrapper:
    vae_lpips: LossColorVAELpipsCfg

class LossColorVAELpips(Loss[LossColorVAELpipsCfg, LossColorVAELpipsCfgWrapper]):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg", verbose=False)
        self._to_loss_device = False
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        autoencoder: FirstStageAdapter,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Get ground-truth appearance.
        gt = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")

        if global_step < self.cfg.apply_after_step or global_step > self.cfg.disable_after_step:
            return torch.tensor(0.0, dtype=torch.float32, device=gt.device)

        if not self._to_loss_device:
            print(f'[INFO] Moving VAE-LPIPS to {gt.device}')
            self.lpips = self.lpips.to(gt.device)
            self._to_loss_device = True

        # Compute LPIPS between ground-truth and recostructed image.
        loss = self.lpips.forward(
            prediction.color_decoded,
            gt,
            normalize=True
        )
        return self.cfg.weight * loss.mean()
