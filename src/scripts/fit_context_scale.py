"""
Fit a per-scene log-affine depth scale (context_oracle / context_rasterized) for an existing
evaluation output tree, so predicted depth can be unprojected into point clouds.

Uses the depth pseudo-labels for context views when available (depth_gt/*.pt), otherwise falls back to
a VGGT pass over context_image_*.png (with sky segmented out via skyseg). See README.md section
"Point clouds from predicted depth" section for the full workflow.

    python -m src.scripts.fit_context_scale --root outputs/<your video eval>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..misc.depth_io import fit_log_affine_scene

VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_CONF_THRESHOLD = 1.0   # matches config/labels.yaml's conf_thres_value

SKYSEG_REPO_ID = "JianyuanWang/skyseg"
SKYSEG_FILENAME = "skyseg.onnx"
SKYSEG_MODEL_URL = f"https://huggingface.co/{SKYSEG_REPO_ID}/resolve/main/{SKYSEG_FILENAME}"

OPACITY_THRESHOLD = 0.5
RAW_RANGE_THRESHOLD = -0.05
FAR_CLIP_MARGIN = 0.99   # rasterized depth saturates at far, so exclude a hair below it too
ILL_CONDITIONED_RATIO = 50.0   # flag a fit if it pushes depth this many camera-baselines away


def reliable_mask(ref, far, base=None, saturating=False):
    """Pixels usable as a scale reference: positive, finite, and within the modeled depth range."""
    far_b = far.reshape(-1, 1, 1)
    m = torch.isfinite(ref) & (ref > 0)
    m &= (ref < FAR_CLIP_MARGIN * far_b) if saturating else (ref <= far_b)
    return m if base is None else (m & base)


def camera_span(extrinsics: torch.Tensor) -> float:
    c = extrinsics[:, :3, 3]
    return max(float((c.max(0).values - c.min(0).values).norm()), 1e-6)


def to_z01(pred: torch.Tensor, z_to_01: bool) -> torch.Tensor:
    return (pred.clamp(-1, 1) + 1) * 0.5 if z_to_01 else pred.clamp(0, 1)


def conditioning(a: float, b: float, z01: torch.Tensor, span: float) -> dict:
    """Depth percentiles this (a, b) produces on these z values, relative to the camera baseline."""
    d = torch.exp(a * z01.flatten()[::7].float() + b)
    d = d[torch.isfinite(d)]
    if not d.numel():
        return {"ill_conditioned": True}
    p95 = float(d.quantile(0.95))
    return {"depth_p50": float(d.quantile(0.50)), "depth_p95": p95,
            "p95_over_baseline": p95 / span,
            "ill_conditioned": bool(p95 / span > ILL_CONDITIONED_RATIO)}


def fit(pred: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor | None, z_to_01: bool):
    """Fit log(reference) ~= a*z + b over valid pixels. Shapes (V,H,W); returns (a, b, coverage)."""
    a, b = fit_log_affine_scene(
        pred[None, :, None],
        reference.clamp_min(1e-6)[None, :, None],
        valid_mask=valid[None, :, None] if valid is not None else None,
        z_to_01=z_to_01,
    )
    cov = float(valid.float().mean()) if valid is not None else 1.0
    return float(a.reshape(())), float(b.reshape(())), cov


def load_vggt_model(device: str, model_url: str):
    """Load VGGT once, the first time a scene actually needs the fallback."""
    from ..labels.vggt.models.vggt import VGGT
    model = VGGT()
    state_dict = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def vggt_context_depth(model, image_paths: list, device: str, conf_thres: float):
    """Predicted depth and a confidence mask for a few images, each resized back to its own native
    resolution so it lines up with the model's predicted depth at the same frames."""
    from ..labels.pseudo_label_utils import run_VGGT
    from ..labels.vggt.utils.load_fn import load_and_preprocess_images_square

    # target_size must be a multiple of VGGT's patch size (14).
    images, coords = load_and_preprocess_images_square([str(p) for p in image_paths], target_size=518)
    images = images.to(device)
    _, _, depth_map, depth_conf = run_VGGT(model, images, dtype=torch.float32,
                                           resolution=images.shape[-1])
    depth_map = torch.from_numpy(depth_map).squeeze(-1)   # (N, S, S[, 1]) -> (N, S, S)
    depth_conf = torch.from_numpy(depth_conf)
    if depth_conf.ndim == 4:
        depth_conf = depth_conf.squeeze(-1)

    depths, masks = [], []
    for i, path in enumerate(image_paths):
        w, h = Image.open(path).size
        x1, y1, x2, y2, _, _ = [int(round(v)) for v in coords[i].tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
        d = depth_map[i, y1:y2, x1:x2][None, None]
        c = depth_conf[i, y1:y2, x1:x2][None, None]
        depths.append(F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0])
        masks.append(F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)[0, 0])
    return torch.stack(depths), torch.stack(masks) >= conf_thres


def load_skyseg_session(model_path: str):
    """Load the skyseg ONNX model, the first time a scene actually needs it."""
    import onnxruntime
    from huggingface_hub import hf_hub_download

    if model_path.startswith("http"):
        model_path = hf_hub_download(repo_id=SKYSEG_REPO_ID, filename=SKYSEG_FILENAME)
    return onnxruntime.InferenceSession(model_path)


def sky_free_mask(session, image_paths: list) -> torch.Tensor:
    """Non-sky mask (True = keep) for a few images, each resized back to its own native resolution.
    Segments sky with the skyseg U-2-Net ONNX model, since VGGT's own confidence does not reliably
    flag it."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    masks = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        x = np.asarray(img.resize((320, 320), Image.BILINEAR), dtype=np.float32)
        x = (x / 255 - mean) / std
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        (out,) = session.run([output_name], {input_name: x})
        out = out.squeeze().astype(np.float32)
        out = (out - out.min()) / (out.max() - out.min() + 1e-8) * 255
        raw = Image.fromarray(out.astype("uint8")).resize((w, h), Image.BILINEAR)
        masks.append(torch.from_numpy(np.array(raw)) < 32)
    return torch.stack(masks)


def fit_all_scenes(
    root: Path,
    *,
    overwrite: bool = False,
    rasterized_depth_all: bool = False,
    nearest_fallback: bool = False,
    context_vggt: bool = True,
    vggt_model_path: str = VGGT_MODEL_URL,
    sky_mask: bool = True,
    skyseg_model_path: str = SKYSEG_MODEL_URL,
    device: str | None = None,
) -> dict:
    """Add context-view scale fits to every scene under an eval output tree, in place. Called by
    the CLI below, and directly from ModelWrapper.on_test_end when test.fit_pcd_scale=true.

    Returns a summary dict of the same counters printed to stdout.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    scenes = sorted(p for p in root.iterdir()
                    if (p / "cam_dict/cameras.torch").exists() and (p / "depth").is_dir())
    n_boot = sum(1 for p in scenes if not (p / "scale_fit.json").exists())
    print(f"[ctx-fit] {len(scenes)} scenes under {root} ({n_boot} without a scale_fit.json yet)")

    done = skipped = no_subset = substituted = 0
    missing_pred = vggt_used = no_reference = 0
    deltas_a, deltas_b, gaps = [], [], []
    vggt_model = None    # loaded lazily, at most once
    skyseg_session = None

    for sd in tqdm(scenes, desc="fitting"):
        fit_path = sd / "scale_fit.json"
        meta = json.loads(fit_path.read_text()) if fit_path.exists() else {"scene": sd.name}
        if "context_oracle" in meta.get("fits", {}) and not overwrite:
            skipped += 1
            continue

        cam = torch.load(sd / "cam_dict/cameras.torch", map_location="cpu")
        ctx = [int(i) for i in cam["context_index"].tolist()]
        tgt = [int(i) for i in cam["target_index"].tolist()]

        # depth exists only for target frames
        missing = [i for i in ctx if i not in tgt]
        if missing and not nearest_fallback:
            no_subset += 1
            continue

        fit_frames = [i if i in tgt else min(tgt, key=lambda t: abs(t - i)) for i in ctx]
        max_gap = max(abs(f - i) for f, i in zip(fit_frames, ctx))
        if max_gap:
            substituted += 1

        pred_paths = [sd / "depth" / f"{i:0>6}.pt" for i in fit_frames]
        if not all(p.exists() for p in pred_paths):
            missing_pred += 1
            continue
        pred = torch.stack([torch.load(p, map_location="cpu").float() for p in pred_paths])

        z_to_01 = meta.get("z_to_01")
        if z_to_01 is None:
            z_to_01 = bool(pred.min() < RAW_RANGE_THRESHOLD)

        meta.setdefault("z_to_01", z_to_01)
        meta.setdefault("convention", "pm1" if z_to_01 else "01")
        meta.setdefault("pred_range", [float(pred.min()), float(pred.max())])

        gaps.append(max_gap)
        provenance = {"frames": fit_frames, "context_frames": ctx,
                      "n_frames": len(ctx), "max_frame_gap": max_gap}

        span = camera_span(cam["extrinsics"].float())
        zc = to_z01(pred, z_to_01)
        fits = meta.setdefault("fits", {})
        far_ctx = cam["far"].float()[[tgt.index(i) for i in fit_frames]]

        # depth_gt if present, else a fresh VGGT pass over the context images
        gt_paths = [sd / "depth_gt" / f"{i:0>6}.pt" for i in fit_frames]
        if all(p.exists() for p in gt_paths):
            gt = torch.stack([torch.load(p, map_location="cpu").float() for p in gt_paths])
            gt_mask = reliable_mask(gt, far_ctx)
            sky_frac = float(((gt > 0) & ~gt_mask).float().mean())
            source_fields = {"source": "depth_pseudo_label", "sky_fraction": sky_frac, "far_clipped": True}
        elif context_vggt and max_gap == 0:
            ctx_img_paths = [sd / f"context_image_{i:0>6}.png" for i in ctx]
            if not all(p.exists() for p in ctx_img_paths):
                no_reference += 1
                continue
            if vggt_model is None:
                print(f"[ctx-fit] loading VGGT ({device}) for the fallback...")
                vggt_model = load_vggt_model(device, vggt_model_path)
            gt, gt_mask = vggt_context_depth(vggt_model, ctx_img_paths, device,
                                             VGGT_CONF_THRESHOLD)
            if sky_mask:
                if skyseg_session is None:
                    print("[ctx-fit] loading skyseg for sky masking...")
                    skyseg_session = load_skyseg_session(skyseg_model_path)
                gt_mask = gt_mask & sky_free_mask(skyseg_session, ctx_img_paths)
            vggt_used += 1
            source_fields = {"source": "vggt_context", "far_clipped": False,
                             "sky_masked": bool(sky_mask)}
        else:
            no_reference += 1
            continue

        a, b, cov = fit(pred, gt, gt_mask, z_to_01)
        fits["context_oracle"] = {"a": a, "b": b, "coverage": cov,
                                  **conditioning(a, b, zc, span), **provenance, **source_fields}

        rast_path = sd / "depth_rast/rasterized.torch"
        if rast_path.exists():
            blob = torch.load(rast_path, map_location="cpu")
            pos = [tgt.index(i) for i in fit_frames]   # fit_frames, not ctx: may be substituted
            rd = blob["depth"].float()[pos]
            rm = reliable_mask(rd, far_ctx, base=blob["mask"].float()[pos] > OPACITY_THRESHOLD,
                               saturating=True)
            if rd.shape == pred.shape and rm.any():
                a2, b2, cov2 = fit(pred, rd, rm, z_to_01)
                fits["context_rasterized"] = {"a": a2, "b": b2, "coverage": cov2,
                                              "far_clipped": True,
                                              **conditioning(a2, b2, zc, span), **provenance}

        if rasterized_depth_all and rast_path.exists():
            far_all = cam["far"].float()
            pa = torch.stack([torch.load(sd / "depth" / f"{i:0>6}.pt", map_location="cpu").float()
                              for i in tgt])
            ra = blob["depth"].float()
            rmask = reliable_mask(ra, far_all,
                                  base=blob["mask"].float() > OPACITY_THRESHOLD, saturating=True)
            if ra.shape == pa.shape and rmask.any():
                a4, b4, c4 = fit(pa, ra, rmask, z_to_01)
                fits["rasterized"] = {"a": a4, "b": b4, "coverage": c4, "far_clipped": True,
                                      "n_frames": len(tgt),
                                      **conditioning(a4, b4, to_z01(pa, z_to_01), span)}

        if "rasterized" in fits:
            deltas_a.append(fits["context_oracle"]["a"] - fits["rasterized"]["a"])
            deltas_b.append(fits["context_oracle"]["b"] - fits["rasterized"]["b"])

        fit_path.write_text(json.dumps(meta, indent=2))
        done += 1

    print(f"\n[ctx-fit] updated={done} skipped={skipped} "
          f"unusable(context not in targets)={no_subset} missing_predicted_depth={missing_pred}")
    print(f"[ctx-fit] context_oracle reference: from existing depth_gt={done - vggt_used}"
          f"   fresh VGGT pass={vggt_used}   unavailable (skipped)={no_reference}")
    if gaps:
        g = torch.tensor(gaps, dtype=torch.float32)
        exact = int((g == 0).sum())
        print(f"[ctx-fit] fits using the ACTUAL context frames: {exact}/{len(gaps)}"
              f"   substituted: {substituted}")
        if substituted:
            far = g[g > 0]
            print(f"[ctx-fit] WARNING substituted frames sit {far.min():.0f}-{far.max():.0f} frames "
                  f"from the context view (median {far.median():.0f}). Only max_frame_gap==0 fits "
                  f"support the 'inference-time only' argument; treat the rest as diagnostics.")
    if deltas_a:
        da = torch.tensor(deltas_a).abs()
        db = torch.tensor(deltas_b).abs()
        print(f"[ctx-fit] |context_oracle - rasterized|  a: median {da.median():.3f} max {da.max():.3f}"
              f"   b: median {db.median():.3f} max {db.max():.3f}")
        print("[ctx-fit] (b is a log-depth offset: 0.10 is ~10% in metric depth, 0.69 is 2x)")

    return {"done": done, "skipped": skipped, "no_subset": no_subset, "substituted": substituted,
            "missing_pred": missing_pred, "vggt_used": vggt_used, "no_reference": no_reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Eval output tree to augment in place")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute context fits for scenes that already have them")
    parser.add_argument("--rasterized-depth-all", action="store_true",
                        help="Also compute the all-target, oracle-free 'rasterized' fit against "
                             "the Gaussian-rasterized depth from export_rasterized_depth.py, for "
                             "comparison against the context-only fits.")
    parser.add_argument("--nearest-fallback", action="store_true",
                        help="When a context view is not among the target views, fit at the "
                             "closest target frame instead of skipping the scene.")
    parser.add_argument("--context-vggt", action=argparse.BooleanOptionalAction, default=True,
                        help="When depth_gt is missing for a scene's context frames, run VGGT over "
                             "its context_image_*.png files instead of skipping the scene "
                             "(default: on). Needs a GPU.")
    parser.add_argument("--vggt-model-path", default=VGGT_MODEL_URL,
                        help="URL or local path to VGGT-1B weights for the fallback.")
    parser.add_argument("--sky-mask", action=argparse.BooleanOptionalAction, default=True,
                        help="When the VGGT fallback runs, also segment sky out of the context "
                             "images and exclude it from the fit (default: on).")
    parser.add_argument("--skyseg-model-path", default=SKYSEG_MODEL_URL,
                        help="URL or local path to the skyseg ONNX weights.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    fit_all_scenes(
        Path(args.root),
        overwrite=args.overwrite,
        rasterized_depth_all=args.rasterized_depth_all,
        nearest_fallback=args.nearest_fallback,
        context_vggt=args.context_vggt,
        vggt_model_path=args.vggt_model_path,
        sky_mask=args.sky_mask,
        skyseg_model_path=args.skyseg_model_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
