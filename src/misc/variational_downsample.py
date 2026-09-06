import torch
import torch.nn.functional as F

def downsample_mu_logvar_mixture(
    mu_hr: torch.Tensor,
    logvar_hr: torch.Tensor,
    factor: int = 8,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample a Gaussian field (mu, logvar) by matching the first two moments
    of the mixture inside each non-overlapping block of size `factor x factor`.

    Args:
        mu_hr:      (B, C, H, W) tensor of means at high resolution.
        logvar_hr:  (B, C, H, W) tensor of log-variances at high resolution.
        factor:     Downsampling factor (e.g., 8 for 1/8 scale latents).
        eps:        Numerical epsilon for stability.

    Returns:
        mu_lr:      (B, C, H/factor, W/factor) downsampled means.
        logvar_lr:  (B, C, H/factor, W/factor) downsampled log-variances.
    """

    # Convert log-variance to variance
    var_hr = torch.exp(logvar_hr)

    # Compute average of means mu_avg = E[mu]
    pool = torch.nn.AvgPool2d(kernel_size=factor, stride=factor)
    mu_lr = pool(mu_hr)

    # Compute average of second moments  E[z**2] = E[std**2 + mean**2]
    second_moment = pool(var_hr + mu_hr**2)

    # Law of total variance: Var = E[z**2] − (E[z])**2
    var_lr = torch.clamp(second_moment - mu_lr**2, min=eps)
    logvar_lr = torch.log(var_lr)

    return mu_lr, logvar_lr