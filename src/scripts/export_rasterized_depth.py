"""
Regenerate the Gaussian-rasterized depth for an existing evaluation output tree, and fit the
log-affine transform that maps the diffusion model predictions from the normalized range [-1,1] to the same scale.

    python -m src.scripts.export_rasterized_depth \
        --experiment re10k_diffusion_release_video_ft \
        --index assets/re10k_evaluation/re10k_extrapolation_top_100_supplement_video.json \
        --num-context-views 2 \
        --out outputs/video_eval_re10k_release_video_ft_step100k \
        --limit 20                     # subset first; drop --limit for the full set

    python -m src.scripts.export_rasterized_depth \
        --experiment dl3dv_diffusion_release_video_ft \
        --index assets/dl3dv_evaluation/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json \
        --num-context-views 4 \
        --out outputs/video_eval_dl3dv_release_video_ft_step100k \
        --limit 20
"""

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize
from tqdm import tqdm

from ..config import load_typed_root_config
from ..dataset.data_module import DataModule, get_data_shim
from ..global_cfg import set_cfg
from ..misc.depth_io import fit_log_affine_scene
from ..misc.step_tracker import StepTracker
from ..model.decoder import get_decoder
from ..model.encoder import get_encoder


RAW_RANGE_THRESHOLD = -0.05
OPACITY_THRESHOLD = 0.5


def build_model(cfg, device):
    """Encoder + rasterizing decoder, with encoder weights from the first-stage checkpoint.

    The first-stage checkpoint holds encoder.* and first_stage.* keys; the decoder is the CUDA
    splatter and has no learnable parameters.
    """
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)

    ckpt_path = Path(cfg.checkpointing.load_first_stage)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"First-stage checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    encoder_sd = {
        k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    print(f"[export] encoder weights from {ckpt_path}")
    print(f"[export]   loaded {len(encoder_sd)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"[export]   WARNING missing keys, first few: {missing[:5]}")

    return encoder.to(device).eval(), decoder.to(device).eval()


def load_predicted_depth(scene_dir: Path, target_index) -> torch.Tensor | None:
    """Stack the per-frame diffusion depth predictions the eval wrote, ordered by target index."""
    frames = []
    for idx in target_index:
        p = scene_dir / "depth" / f"{int(idx):0>6}.pt"
        if not p.exists():
            return None
        frames.append(torch.load(p, map_location="cpu").float())
    return torch.stack(frames)


def fit_against(pred: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor | None, z_to_01: bool):
    """Fit log(reference) ~= a*z + b over valid pixels. Shapes are (V,H,W); returns floats."""
    z = pred[None, :, None]                      # (1,V,1,H,W) as fit_log_affine_scene expects
    d = reference[None, :, None]
    m = valid[None, :, None] if valid is not None else None
    a, b = fit_log_affine_scene(z, d, valid_mask=m, z_to_01=z_to_01)
    return float(a.reshape(())), float(b.reshape(()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="Hydra experiment name (no +experiment= prefix)")
    parser.add_argument("--index", required=True, help="Evaluation view-sampler index JSON")
    parser.add_argument("--out", required=True, help="Existing eval output root to augment in place")
    parser.add_argument("--num-context-views", type=int, required=True, help="2 for RE10K, 4 for DL3DV")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N scenes (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Redo scenes that already have output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_root = Path(args.out)
    if not out_root.is_dir():
        raise FileNotFoundError(f"Eval output root does not exist: {out_root}")

    overrides = [
        f"+experiment={args.experiment}",
        "mode=test",
        "wandb.mode=disabled",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.index_path={args.index}",
        f"dataset.view_sampler.num_context_views={args.num_context_views}",
        "data_loader.test.batch_size=1",
        "checkpointing.load_second_stage=null",
        f"test.output_path={args.out}",
    ]
    with initialize(version_base=None, config_path="../../config"):
        cfg_dict = compose(config_name="main", overrides=overrides)

    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)

    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)

    encoder, decoder = build_model(cfg, device)
    data_shim = get_data_shim(encoder)

    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    loader = data_module.test_dataloader()

    done = skipped = failed = 0
    for batch in tqdm(loader, desc="rasterizing"):
        scene = batch["scene"][0]
        scene_dir = out_root / scene
        if not scene_dir.is_dir():
            skipped += 1
            continue
        if (scene_dir / "scale_fit.json").exists() and not args.overwrite:
            skipped += 1
            continue
        if args.limit and done >= args.limit:
            break

        batch = data_shim(batch)
        batch = {
            k: ({kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                if isinstance(v, dict) else v)
            for k, v in batch.items()
        }
        _, v, _, h, w = batch["target"]["image"].shape

        gaussians = encoder(batch["context"], 0, deterministic=False)
        output = decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=None,
        )

        rast_depth = output.depth[0].float().cpu()               # (V,H,W)
        rast_mask = output.mask[0].float().cpu()                 # (V,H,W) opacity
        target_index = batch["target"]["index"][0].cpu()
        near = batch["target"]["near"][0].float().cpu()          # (V,)

        rast_depth = rast_depth * near[:, None, None]

        (scene_dir / "depth_rast").mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "depth": rast_depth.half(),   # half is plenty: this is only ever a scale reference
                "mask": rast_mask.half(),
                "target_index": target_index,
            },
            scene_dir / "depth_rast/rasterized.torch",
        )

        # Fit the log-affine transform against every reference we have.
        pred = load_predicted_depth(scene_dir, target_index.tolist())
        if pred is None:
            print(f"[export] {scene}: no depth/*.pt on disk, wrote rasterized depth only")
            failed += 1
            continue

        if pred.shape != rast_depth.shape:
            print(f"[export] {scene}: SHAPE MISMATCH pred {tuple(pred.shape)} vs rast "
                  f"{tuple(rast_depth.shape)} -- skipping fit")
            failed += 1
            continue

        z_to_01 = bool(pred.min() < RAW_RANGE_THRESHOLD)

        far_v = batch["target"]["far"][0].float().cpu()[:, None, None]
        valid = (rast_mask > OPACITY_THRESHOLD) & (rast_depth < 0.99 * far_v)
        coverage = float(valid.float().mean())

        fits = {}
        if coverage > 0.01:
            a, b = fit_against(pred, rast_depth.clamp_min(1e-6), valid, z_to_01)
            fits["rasterized"] = {"a": a, "b": b, "coverage": coverage}
        else:
            print(f"[export] {scene}: rasterized coverage {coverage:.4f} too low to fit")

        gt_path = scene_dir / "depth_gt"
        gt_frames = [gt_path / f"{int(i):0>6}.pt" for i in target_index.tolist()]
        if all(p.exists() for p in gt_frames):
            gt = torch.stack([torch.load(p, map_location="cpu").float() for p in gt_frames])
            if gt.shape == pred.shape:
                gt_mask = (gt > 0) & torch.isfinite(gt) & (gt <= far_v)
                a, b = fit_against(pred, gt.clamp_min(1e-6), gt_mask, z_to_01)
                fits["oracle"] = {"a": a, "b": b, "coverage": float(gt_mask.float().mean()),
                                  "sky_fraction": float(((gt > 0) & ~gt_mask).float().mean())}

        with (scene_dir / "scale_fit.json").open("w") as f:
            json.dump(
                {
                    "scene": scene,
                    "z_to_01": z_to_01,
                    "convention": "pm1" if z_to_01 else "01",
                    "rast_units": "world",   # already multiplied by near; see the note in main()
                    "pred_range": [float(pred.min()), float(pred.max())],
                    "rast_range": [float(rast_depth.min()), float(rast_depth.max())],
                    "fits": fits,
                },
                f,
                indent=2,
            )
        done += 1

    print(f"\n[export] done={done} skipped={skipped} failed={failed}")
    print(f"[export] wrote depth_rast/rasterized.torch and scale_fit.json under {out_root}")


if __name__ == "__main__":
    main()
