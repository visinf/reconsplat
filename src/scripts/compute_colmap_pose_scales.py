"""
Per-scene scale that brings the COLMAP camera poses from the chunks into the VGGT pose scale.

    python -m src.scripts.compute_colmap_pose_scales \
        --split test --out assets/re10k_colmap_pose_scales_test.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat

from ..dataset.label_paths import resolve_label_dir


def centres(poses: np.ndarray) -> np.ndarray:
    """Camera centres from the datasets' camera format (w2c in the last 12 entries)."""
    p = torch.as_tensor(np.asarray(poses), dtype=torch.float64)
    w2c = repeat(torch.eye(4, dtype=torch.float64), "h w -> b h w", b=p.shape[0]).clone()
    w2c[:, :3] = rearrange(p[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse()[:, :3, 3].numpy()


def umeyama_scale(src: np.ndarray, dst: np.ndarray):
    """
    Least-squares similarity taking `src` onto `dst`; returns (scale, rmse after alignment).
    """
    sc, dc = src - src.mean(0), dst - dst.mean(0)
    var = (sc ** 2).sum()
    if var < 1e-12:
        return float("nan"), float("nan")
    U, S, Vt = np.linalg.svd(sc.T @ dc)
    d = np.sign(np.linalg.det(U @ Vt))
    S = S.copy()
    S[-1] *= d                                  # keep it a rotation, not a reflection
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    s = S.sum() / var
    rmse = float(np.sqrt(((dc - s * (sc @ R)) ** 2).sum(1).mean()))
    return float(s), rmse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks", type=Path, default=Path("datasets/re10k"))
    p.add_argument("--cameras", type=Path, default=Path("datasets/re10k_vggt_cameras"))
    p.add_argument("--split", default="test")
    p.add_argument("--index", type=Path, default=None,
                   help="evaluation index; restricts the scenes considered")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit-chunks", type=int, default=0)
    args = p.parse_args()

    chunk_dir = args.chunks / args.split
    cam_dir = resolve_label_dir(args.cameras, args.split, "cameras")
    want = None
    if args.index:
        raw = json.loads(args.index.read_text())
        want = {s for s, v in raw.items() if isinstance(v, dict) and v.get("context")}
        print(f"restricting to {len(want)} scenes from {args.index}")

    chunks = sorted(f for f in chunk_dir.iterdir() if f.suffix == ".torch")
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
    print(f"{len(chunks)} chunks under {chunk_dir}")

    scales, rmses = {}, {}
    skipped = {"no_npz": 0, "no_vggt_scene": 0, "degenerate": 0, "short_colmap": 0}
    truncated = 0
    for n, cf in enumerate(chunks, 1):
        npz = cam_dir / cf.name.replace(".torch", ".npz")
        if not npz.exists():
            skipped["no_npz"] += 1
            continue
        cams = np.load(npz)
        chunk = torch.load(cf, map_location="cpu")
        for ex in chunk:
            scene = ex["key"]
            if want is not None and scene not in want:
                continue
            if scene not in cams.files:
                skipped["no_vggt_scene"] += 1
                continue
            col, vgg = centres(ex["cameras"].numpy()), centres(cams[scene])
            if len(col) != len(vgg):
                if len(col) < len(vgg):
                    skipped["short_colmap"] += 1
                    continue
                truncated += 1
                keep = len(vgg)                 
                col, vgg = col[:keep], vgg[:keep]
            s, rmse = umeyama_scale(col, vgg)
            if not np.isfinite(s) or s <= 0:
                skipped["degenerate"] += 1
                continue
            scales[scene], rmses[scene] = s, rmse
        if n % 50 == 0 or n == len(chunks):
            print(f"  {n}/{len(chunks)} chunks, {len(scales)} scenes", flush=True)

    if not scales:
        raise SystemExit("no scenes resolved; check --chunks/--cameras/--split")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scales, indent=1))

    s = np.array(list(scales.values()))
    r = np.array(list(rmses.values()))
    print(f"\nwrote {len(scales)} scenes to {args.out}")
    print(f"  scale  (COLMAP -> VGGT): median {np.median(s):.4f}  "
          f"p10 {np.percentile(s, 10):.4f}  p90 {np.percentile(s, 90):.4f}")
    print(f"  alignment RMSE, VGGT units: median {np.median(r):.4f}  p90 {np.percentile(r, 90):.4f}")
    print(f"  truncated to the common frame prefix: {truncated} scene(s)")
    print(f"  skipped: {skipped}")
    worst = sorted(rmses.items(), key=lambda kv: -kv[1])[:5]
    print("  worst-aligned scenes (COLMAP and VGGT disagree on trajectory shape, not just scale):")
    for k, val in worst:
        print(f"     {k[:24]:26s} RMSE {val:.4f}")
    print("\nRMSE is the residual after the best similarity fit, in VGGT units where a scene spans "
          "~1.\nA large value means COLMAP and VGGT disagree on the trajectory shape for that scene, "
          "not\njust the scale.")


if __name__ == "__main__":
    main()
