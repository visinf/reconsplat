
import torch

from .denoiser import Denoiser
from .mvunet import MultiViewUNetCfg, MultiViewUNet

DENOISER = {
    "mv_unet": MultiViewUNet
}

DenoiserCfg = MultiViewUNetCfg

def get_denoiser(
    denoiser_cfg: DenoiserCfg,
    in_channels: int, 
    out_channels: int,
    image_height: int,
    image_width: int,
    upsample_latents: bool = False
) -> Denoiser:
    if denoiser_cfg.use_prope_encoding:
        return DENOISER[denoiser_cfg.name](denoiser_cfg, in_channels, out_channels, image_height=image_height, image_width=image_width, upsample_latents=upsample_latents)
    return DENOISER[denoiser_cfg.name](denoiser_cfg, in_channels, out_channels, upsample_latents=upsample_latents)


