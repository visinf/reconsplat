from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_vae_color_lpips import LossColorVAELpips, LossColorVAELpipsCfgWrapper
from .loss_vae_depth_charb import LossDepthVAECharbonnier, LossDepthVAECharbonnierCfgWrapper
from .loss_kl import LossFeatureKl, LossFeatureKlCfgWrapper
from .loss_vae_color_l1 import LossColorVAEL1, LossColorVAEL1CfgWrapper
from .loss_vae_color_gan import LossColorVAEGAN, LossColorVAEGANCfgWrapper
from .loss_diffusion import LossDiffusion, LossDiffusionCfgWrapper
from .loss_vae_depth_ssim import LossDepthVAESSIM, LossDepthVAESSIMCfgWrapper
from typing import Union, TypedDict, Optional

class LossVAEColorRecon(TypedDict):
    l1: Optional[LossColorVAEL1] = None
    vae_lpips: Optional[LossColorVAELpips] = None

class LossVAEDepthRecon(TypedDict):
    vae_depth_ssim: Optional[LossDepthVAESSIM] = None
    vae_depth_charb: Optional[LossDepthVAECharbonnier] = None
    
class LossVAE(TypedDict):
    color_recon: LossVAEColorRecon
    depth_recon: LossVAEDepthRecon
    gan: Optional[LossColorVAEGAN] = None
    kl_feature: Optional[LossFeatureKl] = None

class LossAux(TypedDict):
    mse: Optional[LossMse] = None
    lpips: Optional[LossLpips] = None

class LossDiffuser(TypedDict):
    diffusion: Optional[LossDiffusion] = None

# Loss template to use at different training stages.
class LossDict(TypedDict):
    aux_losses: LossAux
    vae_losses: LossVAE
    diff_losses: LossDiffuser

LOSSES = {
    # Aux losses.
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    # VAE (color & depth) losses.
    LossColorVAEL1CfgWrapper: LossColorVAEL1,
    LossColorVAELpipsCfgWrapper: LossColorVAELpips,
    LossColorVAEGANCfgWrapper: LossColorVAEGAN,
    LossDepthVAECharbonnierCfgWrapper: LossDepthVAECharbonnier,
    LossDepthVAESSIMCfgWrapper: LossDepthVAESSIM,
    LossFeatureKlCfgWrapper: LossFeatureKl,
    # Diffusion loss.
    LossDiffusionCfgWrapper: LossDiffusion,
}

LOSS_MAP = {
    'mse': 'aux_losses',
    'lpips': 'aux_losses',
    'l1': 'vae_losses/color_recon',
    'vae_lpips': 'vae_losses/color_recon',
    'gan': 'vae_losses',
    'vae_depth_charb': 'vae_losses/depth_recon',
    'vae_depth_ssim': 'vae_losses/depth_recon',
    'kl_feature': 'vae_losses',
    'diffusion': 'diff_losses'
}

LossCfgWrapper = (
    LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossColorVAEL1CfgWrapper
    | LossColorVAELpipsCfgWrapper
    | LossColorVAEGANCfgWrapper
    | LossDepthVAECharbonnierCfgWrapper
    | LossDepthVAESSIMCfgWrapper
    | LossFeatureKlCfgWrapper
    | LossDiffusionCfgWrapper
)

def get_losses(cfgs: list[LossCfgWrapper]) -> LossDict:
    
    loss_dict = LossDict(
        vae_losses=LossVAE(
            color_recon=LossVAEColorRecon(),
            depth_recon=LossVAEDepthRecon(),
            color_gan=None,
            kl=None,
        ), 
        aux_losses=LossAux(), 
        diff_losses=LossDiffuser()
    )

    for cfg in cfgs:
        loss: Loss = LOSSES[type(cfg)](cfg)
        loss_group = LOSS_MAP[loss.name]
        loss_group_fields = loss_group.split('/')
        if len(loss_group_fields) == 2:
            loss_root, loss_sub = loss_group_fields[:]
            assert isinstance(loss_dict[loss_root], dict)
            loss_dict[loss_root][loss_sub][loss.name] = loss
        else:
            loss_group = loss_group_fields[0]
            loss_dict[loss_group][loss.name] = loss
    return loss_dict