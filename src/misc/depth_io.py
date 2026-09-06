
import argparse
from pathlib import Path
import numpy as np
import zarr
import numcodecs
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from misc.image_io import render_depth_map

# Normalization utils
def normalize_depth(depth, near, far):
    """Normalize depth in [near, far] to [-1, 1]."""
    return 2 * (depth - near) / (far - near) - 1

def denormalize_depth(depth_norm, near, far):
    """Map normalized depth [-1, 1] back to [near, far]."""
    return near + (depth_norm + 1) * (far - near) / 2

# Normalize depth with raw quantiles per scene (batch) - similarly to Marigold. 
def normalize_depth_quantile(
    depth,                       # (B*V,1,H,W)
    B: int,
    V: int,
    norm_min=-1.0,
    norm_max=1.0,
    q=0.02,
    valid_mask=None,             # optional (B*V,1,H,W) bool
    clip=True,
    eps=1e-6,
    mask_exclude_outliers=True, 
):
    # reshape to (B,V,1,H,W)
    d = depth.view(B, V, 1, *depth.shape[-2:])

    if valid_mask is None:
        vm = torch.ones_like(d, dtype=torch.bool)
    else:
        vm = valid_mask.view(B, V, 1, *depth.shape[-2:]).bool()

    # base validity: finite + >0 (Marigold uses >0)
    vm = vm & torch.isfinite(d) & (d > 0)

    # flatten per scene
    d_flat = d.reshape(B, -1)
    vm_flat = vm.reshape(B, -1)

    min_list, max_list = [], []
    for b in range(B):
        vals = d_flat[b][vm_flat[b]]
        if vals.numel() < 16:
            vals = d_flat[b][torch.isfinite(d_flat[b]) & (d_flat[b] > 0)]
        q_lo, q_hi = torch.quantile(vals, torch.tensor([q, 1.0 - q], device=vals.device))
        min_list.append(q_lo)
        max_list.append(q_hi)

    dmin = torch.stack(min_list).view(B,1,1,1,1)
    dmax = torch.stack(max_list).view(B,1,1,1,1)

    # optionally exclude outliers from loss mask too
    vm_outliers = vm
    if mask_exclude_outliers:
        vm_outliers = vm_outliers & (d >= dmin) & (d <= dmax)

    # normalize
    rng = (dmax - dmin).clamp_min(eps)
    dn = (d - dmin) / rng
    dn = dn * (norm_max - norm_min) + norm_min

    if clip:
        dn = dn.clamp(norm_min, norm_max)

    # return per-view tensors
    dn = dn.reshape(B*V, 1, *depth.shape[-2:])
    vm_outliers = vm_outliers.reshape(B*V, 1, *depth.shape[-2:])
    return dn, vm_outliers, (dmin, dmax)

def denormalize_depth_quantile(depth_norm, norm_min=-1.0, norm_max=1.0):
    """Map normalized depth [-1, 1] back to [0, 1]."""
    # cannot go back to "original" raw depth without knowing GT
    # so we treat it as relative depth in [0, 1]
    depth_norm = torch.clip(depth_norm, -1.0, 1.0)
    depth_linear = (depth_norm - norm_min) / (norm_max - norm_min)
    return depth_linear
    
@torch.no_grad()
def fit_log_affine_scene(
    z_ctx,                 # (B,V,1,H,W) model depth in [-1,1] (or any scalar field)
    d_pseudo_ctx,          # (B,V,1,H,W) pseudo depth in meters (or consistent units)
    valid_mask=None,       # (B,V,1,H,W) bool, optional
    z_to_01=True,
    eps=1e-6,
    huber_delta=1.0,
    iters=3,
):
    """
    Fit per-scene (per B) parameters a,b such that log(d_pseudo) ≈ a*z + b.
    Returns a,b shaped (B,1,1,1,1) for broadcasting.
    """
    # Normalize to float32 AND force real fp32 compute for the rest of this function. Both are
    # needed: under Trainer(precision="bf16-mixed"), this whole function runs inside an ambient
    # autocast region, and autocast unconditionally casts matmul-family ops (Aw.T @ Aw, Aw.T @ yw,
    # A @ x below) to bf16 and returns a bf16 result *regardless of the input dtype* -- so even
    # though z_ctx/d_pseudo_ctx are fp32, a pure matmul like `Aw.T @ yw` still comes out bf16, while
    # `Aw.T @ Aw + eps*eye(...)` happens to survive as fp32 only because the elementwise addition
    # afterward promotes it back. That mismatch is exactly what makes torch.linalg.solve below
    # raise "Expected A and B to have the same dtype". Disabling autocast here makes the whole
    # block run in true fp32, which linalg.solve requires.
    z_ctx = z_ctx.float()
    d_pseudo_ctx = d_pseudo_ctx.float()

    with torch.autocast(device_type=z_ctx.device.type, enabled=False):
        return _fit_log_affine_scene_fp32(z_ctx, d_pseudo_ctx, valid_mask, z_to_01, eps, huber_delta, iters)


def _fit_log_affine_scene_fp32(z_ctx, d_pseudo_ctx, valid_mask, z_to_01, eps, huber_delta, iters):
    B, V, _, H, W = z_ctx.shape

    if z_to_01:
        # map [-1,1] -> [0,1], but keep it smooth and bounded
        z = z_ctx.clamp(-1, 1)
        z = (z + 1.0) * 0.5
    else:
        z = z_ctx

    d = d_pseudo_ctx.clamp_min(eps)
    y = torch.log(d)

    if valid_mask is None:
        m = torch.isfinite(z) & torch.isfinite(y) & (d_pseudo_ctx > 0)
    else:
        m = valid_mask & torch.isfinite(z) & torch.isfinite(y) & (d_pseudo_ctx > 0)

    # Flatten per scene
    zf = z.reshape(B, -1)
    yf = y.reshape(B, -1)
    mf = m.reshape(B, -1)

    a_list, b_list = [], []
    for b in range(B):
        zb = zf[b][mf[b]]
        yb = yf[b][mf[b]]

        # fallback if mask too small
        if zb.numel() < 256:
            zb = zf[b][torch.isfinite(zf[b])]
            yb = yf[b][torch.isfinite(yf[b])]

        # Weighted (robust) least squares with a few IRLS steps (Huber)
        # Model: y = a*z + b
        A = torch.stack([zb, torch.ones_like(zb)], dim=1)  # (N,2)
        w = torch.ones((zb.numel(),), device=zb.device, dtype=zb.dtype)

        x = torch.zeros((2,), device=zb.device, dtype=zb.dtype)  # [a,b]
        for _ in range(iters):
            Aw = A * w[:, None]
            yw = yb * w
            # Solve (Aw^T Aw) x = Aw^T yw
            # Add tiny ridge for stability
            ATA = Aw.T @ Aw + 1e-6 * torch.eye(2, device=zb.device, dtype=zb.dtype)
            ATy = Aw.T @ yw
            x = torch.linalg.solve(ATA, ATy)

            # Huber weights
            r = (A @ x - yb).abs()
            w = torch.where(r <= huber_delta, torch.ones_like(r), huber_delta / (r + 1e-12))

        a_list.append(x[0])
        b_list.append(x[1])

    a = torch.stack(a_list).view(B, 1, 1, 1, 1)
    b = torch.stack(b_list).view(B, 1, 1, 1, 1)
    return a, b


@torch.no_grad()
def apply_log_affine_scene(z_any, a, b, z_to_01=True):
    """
    Apply per-scene params to any predicted depth maps.
    z_any: (B,V,1,H,W) or (B,T,1,H,W) etc.
    a,b : (B,1,1,1,1)
    """
    # Normalize to float32 for a predictable output dtype regardless of the caller's precision
    # (e.g. z_any may be bf16 under Trainer(precision="bf16-mixed"), while a/b are always fp32
    # since fit_log_affine_scene now normalizes its own inputs).
    z_any = z_any.float()
    if z_to_01:
        z = z_any.clamp(-1, 1)
        z = (z + 1.0) * 0.5
    else:
        z = z_any
    return torch.exp(a * z + b)

# ----------------------------
# Quantizers (16 bits)
# ----------------------------
def quantize_uint16_per_frame(depth: np.ndarray, use_log: bool = True):
    """
    depth: [N,H,W] float32
    Returns: u16 [N,H,W], d_min [N], d_max [N], valid [N,H,W], use_log (uint8)
    """
    assert depth.ndim == 3, f"Expected [N,H,W], got {depth.shape}"
    d = depth.astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > 0)
    safe = np.where(valid, d, np.nan)
    if use_log:
        safe = np.log(np.clip(safe, 1e-6, None))

    d_min = np.nanmin(safe, axis=(1, 2)).astype(np.float32)
    d_max = np.nanmax(safe, axis=(1, 2)).astype(np.float32)
    rng = np.maximum(d_max - d_min, 1e-12)

    d_min_b = d_min[:, None, None]
    rng_b = rng[:, None, None]
    safe_filled = np.where(valid, safe, d_min_b)
    u = (safe_filled - d_min_b) / rng_b
    u16 = np.round(np.clip(u, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return u16, d_min, d_max, valid, np.uint8(use_log)


def quantize_uint16_per_scene(depth: np.ndarray, use_log: bool = True, p_lo: float = 1.0, p_hi: float = 99.0):
    """
    depth: [N,H,W] float32
    Robust per-scene scaling using percentiles to ignore outliers.
    Returns: u16 [N,H,W], scene_min (scalar), scene_max (scalar), valid [N,H,W], use_log (uint8)
    """
    assert depth.ndim == 3, f"Expected [N,H,W], got {depth.shape}"
    d = depth.astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > 0)
    safe = np.where(valid, d, np.nan)
    if use_log:
        safe = np.log(np.clip(safe, 1e-6, None))

    flat = safe.reshape(-1)
    scene_min = np.nanpercentile(flat, p_lo).astype(np.float32)
    scene_max = np.nanpercentile(flat, p_hi).astype(np.float32)
    rng = max(scene_max - scene_min, 1e-12)

    u = (np.where(valid, safe, scene_min) - scene_min) / rng
    u16 = np.round(np.clip(u, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return u16, scene_min, scene_max, valid, np.uint8(use_log)

# ----------------------------
# Quantizers (N bits)
# ----------------------------

def _prep_safe(depth_f32, use_log):
    d = depth_f32.astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > 0)
    safe = np.where(valid, d, np.nan)
    if use_log:
        safe = np.log(np.clip(safe, 1e-6, None))
    return safe, valid

def quantize_per_frame(depth: np.ndarray, bits: int = 16, use_log: bool = True):
    """depth: [N,H,W] float32 -> (quant [N,H,W] u8/u16, d_min [N], d_max [N], valid [N,H,W], use_log)"""
    assert depth.ndim == 3
    safe, valid = _prep_safe(depth, use_log)
    d_min = np.nanmin(safe, axis=(1,2)).astype(np.float32)
    d_max = np.nanmax(safe, axis=(1,2)).astype(np.float32)
    rng = np.maximum(d_max - d_min, 1e-12)
    safe_filled = np.where(valid, safe, d_min[:,None,None])
    u = (safe_filled - d_min[:,None,None]) / rng[:,None,None]
    if bits == 8:
        q = np.round(np.clip(u,0,1) * 255).astype(np.uint8)
    elif bits == 16:
        q = np.round(np.clip(u,0,1) * 65535).astype(np.uint16)
    else:
        raise ValueError("bits must be 8 or 16")
    return q, d_min, d_max, valid, np.uint8(use_log)

def quantize_per_scene(depth: np.ndarray, bits: int = 16, use_log: bool = True, p_lo: float = 1.0, p_hi: float = 99.0):
    """depth: [N,H,W] -> (quant [N,H,W], scene_min scalar, scene_max scalar, valid [N,H,W], use_log)"""
    assert depth.ndim == 3
    safe, valid = _prep_safe(depth, use_log)
    flat = safe.reshape(-1)
    scene_min = np.nanpercentile(flat, p_lo).astype(np.float32)
    scene_max = np.nanpercentile(flat, p_hi).astype(np.float32)
    rng = max(scene_max - scene_min, 1e-12)
    u = (np.where(valid, safe, scene_min) - scene_min) / rng
    if bits == 8:
        q = np.round(np.clip(u,0,1) * 255).astype(np.uint8)
    elif bits == 16:
        q = np.round(np.clip(u,0,1) * 65535).astype(np.uint16)
    else:
        raise ValueError("bits must be 8 or 16")
    return q, scene_min, scene_max, valid, np.uint8(use_log)


# ----------------------------
# Converter (NPZ chunk -> Zarr)
# ----------------------------
def chunk_npz_to_zarr(
    in_npz: str,
    out_zarr: str,
    mode: str = "per-frame",   # 'per-frame' or 'per-scene'
    bits: int = 8,             # 8 or 16
    use_log: bool = True,
    store_mask: bool = False,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
):
    """
    Reads a .npz chunk with multiple scene arrays [N,H,W].
    Writes a Zarr directory with one group per scene:
      - depth_u8 or depth_u16
      - d_min, d_max (per-frame [N] or per-scene scalars)
      - attrs: {'use_log': 0/1, 'scaling': ..., 'dtype_bits': 8/16}
      - optional 'valid' boolean mask
    """
    znpz = np.load(in_npz, allow_pickle=False)
    root = zarr.open(out_zarr, mode="w")
    # For uint8, BITSHUFFLE still helps; for uint16, BITSHUFFLE is very good
    compressor = numcodecs.Blosc(cname="zstd", clevel=8, shuffle=numcodecs.Blosc.BITSHUFFLE)

    keys = [k for k in znpz.files if isinstance(znpz[k], np.ndarray) and znpz[k].ndim == 3]
    print(f"[INFO] {in_npz} -> {out_zarr} | scenes={len(keys)} | mode={mode} | bits={bits} | log={use_log}")

    for k in keys:
        arr = znpz[k]
        N,H,W = arr.shape
        if mode == "per-frame":
            q, dmin, dmax, valid, logf = quantize_per_frame(arr, bits=bits, use_log=use_log)
        else:
            q, scene_min, scene_max, valid, logf = quantize_per_scene(arr, bits=bits, use_log=use_log, p_lo=p_lo, p_hi=p_hi)

        g = root.create_group(k)
        depth_name = "depth_u8" if bits == 8 else "depth_u16"
        g.create_dataset(depth_name, data=q, chunks=(1,H,W), compressor=compressor)
        g.attrs["use_log"] = int(logf)
        g.attrs["scaling"] = mode
        g.attrs["dtype_bits"] = bits

        if mode == "per-frame":
            g.create_dataset("d_min", data=dmin, chunks=(min(N,64),), compressor=compressor)
            g.create_dataset("d_max", data=dmax, chunks=(min(N,64),), compressor=compressor)
        else:
            g.create_dataset("d_min", data=np.array([scene_min], dtype=np.float32), compressor=compressor)
            g.create_dataset("d_max", data=np.array([scene_max], dtype=np.float32), compressor=compressor)

        if store_mask:
            g.create_dataset("valid", data=valid.astype(bool), chunks=(1,H,W), compressor=compressor)

    print("[OK] done.")

# ----------------------------
# Loader (auto-detect u8/u16 & scaling)
# ----------------------------
def dequantize_zarr_scene(root, scene_key):
    g = root[scene_key]
    bits = int(g.attrs.get("dtype_bits", 16))
    depth_key = "depth_u8" if bits == 8 else "depth_u16"
    q = g[depth_key][:].astype(np.float32)
    use_log = bool(g.attrs.get("use_log", 0))
    scaling = g.attrs.get("scaling", "per-frame")

    # dequantize
    denom = 255.0 if bits == 8 else 65535.0
    u = q / denom
    if scaling == "per-frame":
        dmin = g["d_min"][:].astype(np.float32)  # [N]
        dmax = g["d_max"][:].astype(np.float32)  # [N]
        rng = np.maximum(dmax - dmin, 1e-12)[:,None,None]
        d = u * rng + dmin[:,None,None]
    else:
        dmin = float(g["d_min"][:][0])
        dmax = float(g["d_max"][:][0])
        rng = max(dmax - dmin, 1e-12)
        d = u * rng + dmin

    if use_log:
        d = np.exp(d)

    if "valid" in g:
        v = g["valid"][:]
        d = np.where(v, d, np.nan)
    return d.astype(np.float32)

def load_scene_from_zarr(zarr_path: str, scene_key: str) -> np.ndarray:
    root = zarr.open(zarr_path, mode="r")
    return dequantize_zarr_scene(root, scene_key)

# ----------------------------
# Comparison utilities
# ----------------------------
def compare_npz_vs_zarr(
    npz_path: str,
    zarr_path: str,
    num_scenes: int = 2,
    num_frames: int = 4,
    seed: int | None = None,
    cmap: str = "viridis",
    save_dir: str | None = None,
    show: bool = True,
):
    """
    For a given chunk: randomly pick `num_scenes` scenes and `num_frames` frames each,
    then plot side-by-side NPZ (original) vs Zarr (decoded) depth visualizations.
    """
    rng = np.random.default_rng(seed)
    npz = np.load(npz_path, allow_pickle=False)

    # Only consider keys that are [N,H,W] arrays and also exist as groups in the Zarr
    root = zarr.open(zarr_path, mode="r")
    scene_keys = [
        k for k in npz.files
        if isinstance(npz[k], np.ndarray)
        and npz[k].ndim == 3
        and k in root
    ]
    if len(scene_keys) == 0:
        raise ValueError("No matching [N,H,W] scenes found in both NPZ and Zarr.")

    chosen_scenes = rng.choice(scene_keys, size=min(num_scenes, len(scene_keys)), replace=False)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for sk in chosen_scenes:
        original = npz[sk]              # [N,H,W] float32 (or your original dtype)
        decoded  = load_scene_from_zarr(zarr_path, sk)  # [N,H,W] float32

        N = original.shape[0]
        idxs = rng.choice(N, size=min(num_frames, N), replace=False)
        idxs = sorted(idxs.tolist())

        # One figure per scene: rows = frames, cols = 2 (NPZ vs Zarr)
        fig, axes = plt.subplots(len(idxs), 2, figsize=(10, 2.5 * len(idxs)), squeeze=False)
        fig.suptitle(f"Scene '{sk}' — NPZ (original) vs Zarr (decoded)")

        for r, i in enumerate(idxs):
            depth_npz  = torch.from_numpy(original[i])
            depth_zarr = torch.from_numpy(decoded[i])

            img_npz  = render_depth_map(depth_npz, cmap=cmap).permute(1, 2, 0).numpy()
            img_zarr = render_depth_map(depth_zarr, cmap=cmap).permute(1, 2, 0).numpy()

            axes[r, 0].imshow(img_npz)
            axes[r, 0].set_title(f"NPZ — frame {i}")
            axes[r, 1].imshow(img_zarr)
            axes[r, 1].set_title(f"Zarr — frame {i}")

            for c in (0, 1):
                axes[r, c].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir:
            out_png = Path(save_dir) / f"{Path(npz_path).stem}_{sk}_compare.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            print(f"[Saved] {out_png}")

        if show:
            plt.show()
        else:
            plt.close(fig)

# ----------------------------
# Quick quality check (NPZ vs decoded)
# ----------------------------
def sample_metrics_npz_vs_zarr(npz_path: str, zarr_path: str, scenes=2, frames=4, seed=0):
    rng = np.random.default_rng(seed)
    npz = np.load(npz_path, allow_pickle=False)
    root = zarr.open(zarr_path, mode="r")
    scene_keys = [k for k in npz.files if isinstance(npz[k], np.ndarray) and npz[k].ndim==3 and k in root]
    if not scene_keys:
        raise ValueError("No matching scenes in NPZ and Zarr")
    scene_sel = rng.choice(scene_keys, size=min(scenes, len(scene_keys)), replace=False)
    out = []
    for sk in scene_sel:
        a = npz[sk].astype(np.float32)           # original
        b = load_scene_from_zarr(zarr_path, sk)  # decoded
        N = a.shape[0]
        frame_sel = sorted(rng.choice(N, size=min(frames, N), replace=False).tolist())
        for i in frame_sel:
            gt = a[i]
            pr = b[i]
            mask = np.isfinite(gt) & (gt > 0)
            if mask.sum() == 0:
                continue
            diff = pr[mask] - gt[mask]
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            rel = float(np.mean(np.abs(diff) / np.maximum(gt[mask], 1e-6)))
            out.append((sk, i, mae, rmse, rel))
    return out