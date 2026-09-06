from dataclasses import dataclass
from jaxtyping import Float
from typing import Optional
from torch import Tensor
from einops import rearrange

from .loss import Loss
from ..dataset.types import BatchedExample
from ..model.types import Gaussians
from ..model.decoder.decoder import DecoderOutput
from ..model.first_stage.adapter import FirstStageAdapter
import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_d_loss(real_logits, fake_logits):
    # real: max(0, 1 - D(x))
    loss_real = F.relu(1.0 - real_logits).mean()
    # fake: max(0, 1 + D(G(z)))
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return (loss_real, loss_fake)

def hinge_g_loss(fake_logits):
    # generator tries to maximize D(G(z))  => minimize -D(G(z))
    return -fake_logits.mean()

@dataclass
class LossColorVAEGANCfg:
    g_weight: float = 0.75                      # weight for generator term
    d_weight: float = 1.0                       # weight for discriminator term
    apply_after_step: Optional[int] = 0         # warm-up before enabling GAN loss
    disable_after_step: Optional[int] = int(1e7)

@dataclass
class LossColorVAEGANCfgWrapper:
    gan: LossColorVAEGANCfg

class LossColorVAEGAN(Loss[LossColorVAEGANCfg, LossColorVAEGANCfgWrapper]):
    """
    Generator-side adversarial loss on VAE reconstructions.
    - Call this in your generator step to compute L_GAN (to be added to recon + LPIPS).
    - Use the companion discriminator step (below) to train D.
    """
    def _resize_like(self, feats, h_full, w_full):
        h_feat, w_feat = feats.shape[2:]
        if h_feat != h_full // 8 or w_feat != w_full // 8:
            feats = F.interpolate(
                feats, size=(h_full // 8, w_full // 8),
                mode="bilinear", align_corners=False, antialias=True
            )
        return feats

    def forward(
        self,
        prediction: DecoderOutput,               
        batch: BatchedExample,
        gaussians: Gaussians,
        autoencoder: FirstStageAdapter,
        recon_loss: Tensor,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # turn off before apply_after_step
        if global_step < self.cfg.apply_after_step or global_step > self.cfg.disable_after_step:
            return torch.tensor(0.0, device=prediction.color_decoded.device, dtype=torch.float32)

        # discriminator logits on fake (patch map)
        fake_logits = autoencoder.gan_discriminator(prediction.color_decoded)
        g_loss = hinge_g_loss(fake_logits)

        last_layer_weights = autoencoder.last_layer_weights()
        # compute adaptive weight based on the magnitudes of g_loss and recon_loss
        recon_grads = torch.autograd.grad(recon_loss, last_layer_weights, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weights, retain_graph=True)[0]
        lam = torch.linalg.vector_norm(recon_grads) / (torch.linalg.vector_norm(g_grads) + 1e-6)
        lam = torch.clamp(lam, min=0, max=1).detach()

        return self.cfg.g_weight * lam * g_loss

    def discriminator_step_loss(
        self,
        prediction: DecoderOutput,               
        batch: BatchedExample,
        autoencoder: FirstStageAdapter,
        global_step: int,
    ):
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0.0, device=prediction.color_decoded.device, dtype=torch.float32)

        # Detach prediction / fake instance to compute discriminator term.
        fake = prediction.color_decoded.detach()
        real = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")

        # compute logits
        real_logits = autoencoder.gan_discriminator(real)
        fake_logits = autoencoder.gan_discriminator(fake)
        loss_real, loss_fake = hinge_d_loss(real_logits, fake_logits)
        return self.cfg.d_weight/2 * loss_fake + self.cfg.d_weight/2 * loss_real 

    def is_adversarial_active(self, global_step: int):
        return global_step > self.cfg.apply_after_step
