import numpy as np
import torch
import math
from dataclasses import dataclass, asdict
from typing import Literal, Union, Any
from diffusers import DDIMScheduler, DDPMScheduler

from .ddim import DDIMSchedulerCfg
from .ddpm import DDPMSchedulerCfg

@dataclass
class SchedulerCfg:
    name: Literal["ddim", "ddpm"]
    num_train_timesteps: int
    num_inference_steps: int
    prediction_type: str 
    pretrained_from: str | None
    shift_schedule: bool
    kwargs: dict[str, Any]
    shift_by_number_of_views: bool = True
    shift_lambda: float = 0.0

SCHEDULER = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}


@torch.no_grad()
def shift_scheduler_logsnr(
    scheduler: DDPMScheduler | DDIMScheduler,
    N: int,
    *,
    direction: str = "more_noise",   # "more_noise" = logSNR - log(N); "less_noise" = logSNR + log(N)
    eps: float = 1e-12,
    shift_by_number_of_views: bool = True,
    shift_lambda: float = 0.0,
) -> DDPMScheduler | DDIMScheduler:
    """
    Shift the logSNR of a DDIM/DDPM-style scheduler by +/- log(N) and update alphas/betas accordingly.
    
    Parameters
    ----------
    scheduler : DDPMScheduler | DDIMScheduler
        A HuggingFace diffusers scheduler instance.
    N : int
        Number of jointly modeled targets (e.g., N views).
    direction : str
        "more_noise" or "less_noise".
    eps : float
        Numerical clamp for alpha_bar in (0, 1).

    Returns
    -------
    scheduler : DDPMScheduler | DDIMScheduler
        The same scheduler, modified in-place and returned for convenience.
    """
    if N <= 1:
        return scheduler

    if direction not in {"more_noise", "less_noise"}:
        raise ValueError(f"direction must be 'more_noise' or 'less_noise', got: {direction}")

    # alpha_bar = alphas_cumprod
    ab = scheduler.alphas_cumprod.detach().clone().float()
    ab = ab.clamp(eps, 1.0 - eps)

    # logSNR = log(ab/(1-ab)) = log(ab) - log(1-ab)
    logsnr = torch.log(ab) - torch.log1p(-ab)

    if shift_by_number_of_views:
        shift = math.log(float(N))
        if direction == "more_noise":
            logsnr_shifted = logsnr - shift
        else:  # "less_noise"
            logsnr_shifted = logsnr + shift
    else:
        logsnr_shifted = lognsr - shift_lambda

    # back: ab' = sigmoid(logSNR')
    ab_shifted = torch.sigmoid(logsnr_shifted).to(scheduler.alphas_cumprod.dtype)

    # Recompute per-step alphas and betas so they are consistent with new cumulative products.
    # For t=0: alpha_0 = ab_0
    # For t>0: alpha_t = ab_t / ab_{t-1}
    alphas = torch.empty_like(ab_shifted)
    alphas[0] = ab_shifted[0]
    alphas[1:] = (ab_shifted[1:] / ab_shifted[:-1]).clamp(eps, 1.0)  # keep sane
    betas = (1.0 - alphas).clamp(0.0, 1.0)

    # Write back (keep original device)
    device = scheduler.alphas_cumprod.device
    scheduler.alphas_cumprod = ab_shifted.to(device)
    scheduler.alphas = alphas.to(device)
    scheduler.betas = betas.to(device)

    # Some schedulers cache this; DDIMScheduler uses it in a few places.
    if hasattr(scheduler, "final_alpha_cumprod"):
        scheduler.final_alpha_cumprod = scheduler.alphas_cumprod[0]

    return scheduler

def _parse_cfg(cfg: SchedulerCfg):
    kwargs = dict(cfg.kwargs)
    # Fix trained_betas type if needed
    if isinstance(kwargs.get("trained_betas"), list):
        kwargs["trained_betas"] = np.array(kwargs["trained_betas"])
    return kwargs

def get_scheduler(
    cfg: SchedulerCfg
) -> Union[DDPMScheduler, DDIMScheduler]:
     
    if cfg.pretrained_from is None:
        return SCHEDULER[cfg.name](**(_parse_cfg(cfg)))
    else:
        return SCHEDULER[cfg.name].from_pretrained(cfg.pretrained_from, subfolder="scheduler")
