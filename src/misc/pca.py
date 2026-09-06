import torch
import numpy as np
from sklearn.decomposition import PCA

@torch.no_grad()
def pca_rgb_per_view(
    feat: torch.Tensor,            # (1, V, C, H, W)  (your: (1, 8, D, H, W))
    max_samples: int = 200_000,     # for fitting PCA
    seed: int = 0,
    robust_q: float = 0.01,         # clip [q, 1-q] per channel
    chunk: int = 2_000_000,
    device_for_sklearn: str = "cpu",
):
    assert feat.ndim == 5 and feat.shape[0] == 1
    B, V, C, H, W = feat.shape

    # (N, C) where N = V*H*W
    x = feat[0].permute(0, 2, 3, 1).reshape(-1, C)  # (V,H,W,C)->(N,C)
    N = x.shape[0]

    x_cpu = x.detach().to(device_for_sklearn)

    # Fit PCA on a random subset for speed
    g = torch.Generator(device=x_cpu.device).manual_seed(seed)
    if N > max_samples:
        idx = torch.randperm(N, generator=g, device=x_cpu.device)[:max_samples]
        x_fit = x_cpu[idx]
    else:
        x_fit = x_cpu

    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    pca.fit(x_fit.numpy())

    # Transform all points, chunked
    out = np.empty((N, 3), dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out[s:e] = pca.transform(x_cpu[s:e].numpy()).astype(np.float32)

    # Robust normalize globally (shared across views) so colors are comparable
    lo = np.quantile(out, robust_q, axis=0)
    hi = np.quantile(out, 1.0 - robust_q, axis=0)
    out = np.clip((out - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    rgb = torch.from_numpy(out).view(V, H, W, 3)  # (V,H,W,3) in [0,1]
    return rgb, pca

@torch.no_grad()
def pca_to_rgb(x: torch.Tensor, eps: float = 1e-6):
    """
    x: (B, V, C, H, W) float tensor
    returns:
      rgb_uint8: (B, V, 3, H, W) uint8 in [0,255]
      rgb_float: (B, V, 3, H, W) float in [0,1]
    """
    assert x.ndim == 5, f"expected (B,V,C,H,W), got {x.shape}"
    B, V, C, H, W = x.shape

    # (N, C) where N = B*V*H*W
    X = x.permute(0, 1, 3, 4, 2).reshape(-1, C).float()

    # center
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    # optional: normalize per-dim to reduce dominance of high-variance channels
    std = Xc.std(dim=0, keepdim=True).clamp_min(eps)
    Xn = Xc / std

    # PCA via SVD: Xn = U S Vh, principal directions are rows of Vh
    # Using the economy SVD
    U, S, Vh = torch.linalg.svd(Xn, full_matrices=False)
    P = Vh[:3].T  # (C, 3)

    Y = Xn @ P    # (N, 3)

    # reshape back to (B,V,3,H,W)
    rgb = Y.view(B, V, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()

    # map to displayable ranges per-image (per B,V) and return both float + uint8
    rgb_flat = rgb.view(B, V, 3, -1)
    mn = rgb_flat.amin(dim=-1, keepdim=True)
    mx = rgb_flat.amax(dim=-1, keepdim=True)
    rgb01 = ((rgb_flat - mn) / (mx - mn).clamp_min(eps)).view(B, V, 3, H, W)

    rgb_uint8 = (rgb01 * 255.0).round().clamp(0, 255).to(torch.uint8)
    return rgb_uint8, rgb01

@torch.no_grad()
def fit_pca_projection(x: torch.Tensor, k: int = 3, eps: float = 1e-6):
    """
    Fit PCA on x and return parameters to reuse on other tensors.
    x: (B,V,C,H,W)
    returns dict with mean, std, P (projection matrix C->k)
    """
    B, V, C, H, W = x.shape
    X = x.permute(0, 1, 3, 4, 2).reshape(-1, C).float()  # (N,C)

    mean = X.mean(dim=0, keepdim=True)        # (1,C)
    Xc = X - mean

    std = Xc.std(dim=0, keepdim=True).clamp_min(eps)  # (1,C)
    Xn = Xc / std

    # PCA via SVD
    U, S, Vh = torch.linalg.svd(Xn, full_matrices=False)
    P = Vh[:k].T.contiguous()  # (C,k)

    return {"mean": mean, "std": std, "P": P}

@torch.no_grad()
def apply_pca_projection(x: torch.Tensor, pca: dict, eps: float = 1e-6):
    """
    Apply a previously fit PCA projection to x.
    x: (B,V,C,H,W)
    returns y: (B,V,k,H,W) float
    """
    mean, std, P = pca["mean"], pca["std"], pca["P"]
    B, V, C, H, W = x.shape
    assert mean.shape[-1] == C and P.shape[0] == C, "Channel dim mismatch."

    X = x.permute(0, 1, 3, 4, 2).reshape(-1, C).float()
    Xn = (X - mean) / std.clamp_min(eps)
    Y = Xn @ P  # (N,k)

    y = Y.view(B, V, H, W, P.shape[1]).permute(0, 1, 4, 2, 3).contiguous()
    return y

@torch.no_grad()
def normalize_to_01_per_image(y: torch.Tensor, eps: float = 1e-6):
    """
    y: (B,V,3,H,W) -> (B,V,3,H,W) in [0,1], per (B,V) for nicer visuals
    """
    B, V, C, H, W = y.shape
    flat = y.view(B, V, C, -1)
    mn = flat.amin(dim=-1, keepdim=True)
    mx = flat.amax(dim=-1, keepdim=True)
    return ((flat - mn) / (mx - mn).clamp_min(eps)).view(B, V, C, H, W)
