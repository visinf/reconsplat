#!/usr/bin/env python3
# quantize_depth.py

# Script to quantize depth (float32 -> uint16) and store scene depth into a Zarr directory.

import os, sys, argparse
# --- Safe defaults for HPC / multi-proc ---
os.environ.setdefault("MPLBACKEND", "Agg")         # headless plotting
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")       # 1 thread per process for Blosc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from misc.depth_io import  chunk_npz_to_zarr, sample_metrics_npz_vs_zarr, compare_npz_vs_zarr
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def process_one_chunk(
    npz_path: str,
    zarr_path: str,
    mode: str,
    bits: int,
    use_log: bool,
    store_mask: bool,
    p_lo: float,
    p_hi: float,
    do_metrics: bool,
    do_compare: bool,
    compare_save_dir: str | None,
    seed_base: int,
):
    """
    Worker function: converts one NPZ chunk to Zarr, runs metrics and (optionally) saves comparisons.
    Returns a short summary string.
    """
    npz_path = Path(npz_path)
    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    if compare_save_dir:
        Path(compare_save_dir).mkdir(parents=True, exist_ok=True)

    # Convert / compress NPZ -> Zarr
    chunk_npz_to_zarr(
        in_npz=str(npz_path),
        out_zarr=str(zarr_path),
        mode=mode,
        bits=bits,
        use_log=use_log,
        store_mask=store_mask,
        p_lo=p_lo,
        p_hi=p_hi,
    )

    lines = [f"[OK] {npz_path.name} -> {zarr_path.name} ({mode}, {bits}-bit, log={use_log})"]

    # Metrics on a small random sample
    if do_metrics:
        rows = sample_metrics_npz_vs_zarr(
            npz_path=str(npz_path),
            zarr_path=str(zarr_path),
            scenes=2,
            frames=4,
            seed=seed_base,
        )
        for sk, i, mae, rmse, rel in rows:
            lines.append(f"{sk} frame {i:>3} | MAE={mae:.4f} m | RMSE={rmse:.4f} m | REL={rel:.3%}")

    # Qualitative comparison (saved PNGs if save_dir provided)
    if do_compare:
        compare_npz_vs_zarr(
            npz_path=str(npz_path),
            zarr_path=str(zarr_path),
            num_scenes=1,
            num_frames=4,
            seed=seed_base,
            cmap="viridis",
            save_dir=compare_save_dir,
            show=False,  # don't open windows
        )
        if compare_save_dir:
            lines.append(f"[Saved] comparisons -> {compare_save_dir}")

    return "\n".join(lines)


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_npz", type=str, help="Path to folder containing NPZ chunks (with subfolders train/ test)")
    ap.add_argument("output_zarr", type=str, help="Output folder where Zarr directories will be written (mirrors train/ test)")
    ap.add_argument("--mode", choices=["per-frame", "per-scene"], default="per-frame",
                    help="Quantization scaling mode")
    ap.add_argument("--bits", type=int, choices=[8, 16], default=8,
                    help="Quantization precision (8 or 16). 8-bit is smaller; 16-bit higher fidelity.")
    ap.add_argument("--no-log", action="store_true", help="Disable log-depth quantization")
    ap.add_argument("--store-mask", action="store_true", help="Store validity mask (bool)")
    ap.add_argument("--p-lo", type=float, default=1.0, help="Lower percentile for per-scene mode")
    ap.add_argument("--p-hi", type=float, default=99.0, help="Upper percentile for per-scene mode")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="Number of parallel worker processes")
    ap.add_argument("--stages", type=str, nargs="+", default=["train", "test"],
                    help="Which subfolders to process (default: train test)")
    ap.add_argument("--metrics", action="store_true", help="Compute sample metrics after each conversion")
    ap.add_argument("--compare", action="store_true", help="Save qualitative comparisons (PNG)")
    ap.add_argument("--compare-dirname", type=str, default="comparisons",
                    help="Subfolder name (under each stage) to save comparison PNGs")
    ap.add_argument("--seed", type=int, default=42, help="Base seed used in sampling for metrics/plots")

    args = ap.parse_args()

    input_root = Path(args.input_npz)
    output_root = Path(args.output_zarr)
    output_root.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for stage in args.stages:
        in_stage = input_root / stage
        out_stage = output_root / stage
        out_stage.mkdir(parents=True, exist_ok=True)

        compare_dir = (out_stage / args.compare_dirname) if args.compare else None

        for npz_path in sorted(in_stage.glob("*.npz")):
            zarr_path = out_stage / f"{npz_path.stem}.zarr"
            tasks.append((
                str(npz_path),
                str(zarr_path),
                args.mode,
                int(args.bits),
                (not args.no_log),
                bool(args.store_mask),
                float(args.p_lo),
                float(args.p_hi),
                bool(args.metrics),
                bool(args.compare),
                (str(compare_dir) if compare_dir else None),
            ))

    if not tasks:
        print("No NPZ chunks found. Check --stages and input folder.")
        return

    print(f"[INFO] Found {len(tasks)} chunks across stages: {', '.join(args.stages)}")
    print(f"[INFO] Running with {args.workers} workers | mode={args.mode} | bits={args.bits} | log={not args.no_log}")

    # Partial worker with constant seed offset per task index (for variety)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for idx, t in enumerate(tasks):
            seed_base = args.seed + idx
            fut = ex.submit(process_one_chunk, *t, seed_base)
            futures[fut] = Path(t[0]).name

        # Collect results as they finish
        completed = 0
        for fut in as_completed(futures):
            chunk_name = futures[fut]
            try:
                msg = fut.result()
                print(msg)
            except Exception as e:
                print(f"[ERROR] {chunk_name}: {e}", file=sys.stderr)
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                print(f"[PROGRESS] {completed}/{len(tasks)} done")

    print("[ALL DONE]")

if __name__ == "__main__":
    main()