from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from DISTS_pytorch import DISTS

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    eps: float = 1e-10,
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * (mse + eps).log10()

@torch.no_grad()
def compute_mse(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b h w -> b", "mean")
    return mse

@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().float().cpu().numpy(),
            hat.detach().float().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@cache
def get_dists(device: torch.device) -> DISTS:
    return DISTS().to(device)

@torch.no_grad()
def compute_dists(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_dists(predicted.device).forward(
        ground_truth, predicted, require_grad=False
    )
    if value.ndim == 0:
        return value.unsqueeze(0)
    return value

# depth estimation metrics
@torch.no_grad()
def abs_error(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height widht"],
) -> Float[Tensor, "batch"]:
    # print(ground_truth.shape, predicted.shape) # (90, 256,448)
    delta = ((predicted - ground_truth).abs() / (ground_truth + 1e-8)).mean(dim=(1, 2))
    return delta


@torch.no_grad()
def log_rmse_error(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height widht"],
) -> Float[Tensor, "batch"]:
    delta = (
        ((predicted.clamp(min=1e-8).log() - ground_truth.clamp(min=1e-8).log()) ** 2)
        .mean(dim=(1, 2))
        .sqrt()
    )
    return delta


@torch.no_grad()
def rmse_error(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height widht"],
) -> Float[Tensor, "batch"]:
    delta = ((predicted - ground_truth) ** 2).mean(dim=(1, 2)).sqrt()
    return delta

@torch.no_grad()
def delta(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height widht"],
):
    ratio = torch.maximum(predicted / ground_truth, ground_truth / predicted)
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < 1.25**2).float().mean()
    delta3 = (ratio < 1.25**3).float().mean()
    return delta1, delta2, delta3

@torch.no_grad()
def SiLO(ground_truth, predicted):
    log_diff = torch.log(predicted + 1e-8) - torch.log(ground_truth + 1e-8)

    silo = torch.sqrt((log_diff**2).mean(dim=(1, 2)) - (log_diff.mean(dim=(1, 2)) ** 2))
    return silo
