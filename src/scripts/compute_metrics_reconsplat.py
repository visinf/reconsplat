"""
Score color and depth metrics for a novel-view depth synthesis eval tree.

    python -m src.scripts.compute_metrics_reconsplat --root outputs/<your scannet eval>
"""

import argparse
from pathlib import Path

import torch
from tabulate import tabulate

from ..evaluation.metrics import (
    SiLO,
    abs_error,
    compute_dists,
    compute_lpips,
    compute_psnr,
    compute_ssim,
    delta,
    log_rmse_error,
    rmse_error,
)
from ..misc.image_io import load_image

METRICS = ["psnr", "ssim", "lpips", "dists",
           "abs_error", "rmse_error", "log_rmse_error", "delta1", "delta2", "delta3", "silog"]


def scale_shift(depth: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
    """Per-scene least-squares affine fit of predicted depth against reference."""
    pred_flat = depth.reshape(-1, 1)
    gt_flat = depth_gt.reshape(-1, 1)
    A = torch.cat([pred_flat, torch.ones_like(pred_flat)], dim=1)
    scale, shift = torch.linalg.lstsq(A, gt_flat).solution
    return depth * scale + shift


def score_scene(scene_dir: Path, device: str) -> dict:
    color = torch.stack([load_image(f) for f in sorted((scene_dir / "color").glob("*.png"))]).to(device)
    color_gt = torch.stack([load_image(f) for f in sorted((scene_dir / "color_gt").glob("*.png"))]).to(device)
    depth = torch.stack([torch.load(f, map_location="cpu")
                        for f in sorted((scene_dir / "depth").glob("*.pt"))]).to(device)
    depth_gt = torch.stack([torch.load(f, map_location="cpu")
                            for f in sorted((scene_dir / "depth_gt").glob("*.pt"))]).to(device)
    depth_gt = depth_gt / 1000   # ScanNet++'s iPhone depth is stored in millimeters

    depth_aligned = scale_shift(depth, depth_gt)
    d1, d2, d3 = delta(depth_gt, depth_aligned)
    return {
        "psnr": compute_psnr(color_gt, color).mean().item(),
        "ssim": compute_ssim(color_gt, color).mean().item(),
        "lpips": compute_lpips(color_gt, color).mean().item(),
        "dists": compute_dists(color_gt, color).mean().item(),
        "abs_error": abs_error(depth_gt, depth_aligned).mean().item(),
        "rmse_error": rmse_error(depth_gt, depth_aligned).mean().item(),
        "log_rmse_error": log_rmse_error(depth_gt, depth_aligned).mean().item(),
        "delta1": d1.item(), "delta2": d2.item(), "delta3": d3.item(),
        # Use the raw (not scale/shift-aligned) prediction: SiLog is itself invariant to an
        # unknown global scale, so aligning first would be redundant.
        "silog": SiLO(depth_gt, depth).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="Eval output tree: <root>/<scene>/{color,color_gt,depth,depth_gt}")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(args.root)
    scenes = sorted(p for p in root.iterdir() if (p / "depth_gt").is_dir())
    if not scenes:
        raise FileNotFoundError(f"No scene with a depth_gt/ folder under {root}")

    totals = {k: [] for k in METRICS}
    for scene_dir in scenes:
        for k, v in score_scene(scene_dir, args.device).items():
            totals[k].append(v)

    rows = [[k, f"{sum(v) / len(v):.4f}"] for k, v in totals.items()]
    print(f"{len(scenes)} scene(s) under {root}\n")
    print(tabulate(rows, headers=["metric", "average"]))


if __name__ == "__main__":
    main()
