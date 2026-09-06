"""
Compute FID / KID for prediction folders against pre-registered clean-fid
"custom" dataset stats.

Usage:
    python -m src.scripts.fid_score_predictions                 # run all experiments
    python -m src.scripts.fid_score_predictions dl3dv           # run just one
    python -m src.scripts.fid_score_predictions --list          # list available experiments
"""

import argparse
import json
import os

from cleanfid import fid

EXPERIMENTS = {
    "re10k_main_reconsplat": {
        "description": "RE10k main results: ReconSplat (interpolation + extrapolation).",
        "output_path": "outputs/fid_stats_reconsplat_re10k_new.json",
        "groups": {
            "main": {
                "dataset_name_template": "re10k-test-{method}",
                "compute_kid": True,
                "flatten": True,
                "paths": {
                    "interpolation": "/path/to/predictions/reconsplat_re10k_interpolaton_fid_images",
                    "extrapolation": "/path/to/predictions/reconsplat_re10k_extrapolation_fid_images",
                },
            },
        },
    },
    "re10k_main_mvsplat": {
        "description": "RE10k main results: MVSplat baseline (interpolation + extrapolation).",
        "output_path": "outputs/fid_stats_mvsplat_re10k_new.json",
        "groups": {
            "main": {
                "dataset_name_template": "re10k-test-{method}",
                "compute_kid": True,
                "flatten": True,
                "paths": {
                    "interpolation": "/path/to/predictions/baselines_asset_ours/mvsplat_interpolation_fid_images_flattened",
                    "extrapolation": "/path/to/predictions/baselines_asset_ours/mvsplat_extrapolation_fid_images_flattened",
                },
            },
        },
    },
    "ablations": {
        "description": "RE10k extrapolation ablations (no_mr_noise / no_depth / no_gaussians / no_ft).",
        "output_path": "outputs/reconsplat_fid_stats_ablations.json",
        "groups": {
            "extrapolation": {
                "dataset_name": "re10k-test-extrapolation",
                "compute_kid": False,
                "paths": {
                    "no_mr_noise": "/path/to/predictions/re10_extrapolation_no_mr_noise_fid_images",
                    "no_depth": "/path/to/predictions/re10_extrapolation_no_depth_fid_images",
                    "no_gaussians": "/path/to/predictions/re10_extrapolation_no_gaussians_fid_images",
                    "no_ft": "/path/to/predictions/re10_extrapolation_no_ft_fid_images",
                },
            },
        },
    },
    "dl3dv": {
        "description": "DL3DV main results (n=150 and n=300, cfg=3).",
        "output_path": "outputs/fid_stats_reconsplat_eccv_dl3dv.json",
        "groups": {
            "n150": {
                "dataset_name": "dl3dv-test-n150",
                "compute_kid": True,
                "paths": {
                    "cfg3": "outputs/reconsplat_prope_vpred_eval_dl3dv_n150_as_video_cfg3_upsample_tmp/samples",
                },
            },
            "n300": {
                "dataset_name": "dl3dv-test-n300",
                "compute_kid": True,
                "paths": {
                    "cfg3": "outputs/reconsplat_prope_vpred_eval_dl3dv_n300_as_video_cfg3_upsample_tmp/samples",
                },
            },
        },
    },
    "dl3dv_exps": {
        "description": "DL3DV n=150 context-view-count ablation (6/8 views) and RE10k->DL3DV cross-dataset generalization.",
        "output_path": "outputs/fid_stats_reconsplat_dl3dv_eccv_supplement.json",
        "groups": {
            "n150": {
                "dataset_name": "dl3dv-test-n150",
                "compute_kid": True,
                "paths": {
                    "reconsplat_6view": "outputs/reconsplat_prope_vpred_eval_dl3dv_n150_6v_cfg3_upsample_tmp/samples",
                    "reconsplat_8view": "outputs/reconsplat_prope_vpred_eval_dl3dv_n150_8v_cfg3_upsample_tmp/samples",
                },
            },
            "cross": {
                "dataset_name": "dl3dv-from-re10k",
                "compute_kid": True,
                "paths": {
                    "reconsplat_from_re10k": "outputs/reconsplat_prope_vpred_eval_dl3dv_from_re10k_2views/samples",
                },
            },
        },
    },
    "dl3dv_supplement": {
        "description": "DL3DV n=150 classifier-free-guidance (CFG) ablation, for the supplement.",
        "output_path": "outputs/dl3dv_fid_reconsplat_supplement.json",
        "groups": {
            "n150": {
                "dataset_name": "dl3dv-test-n150",
                "compute_kid": False,
                "paths": {
                    "cfg1": "/path/to/predictions/reconsplat_dl3dv_n150_fid_images_cfg1",
                    "cfg5": "/path/to/predictions/reconsplat_dl3dv_n150_fid_images_cfg5",
                    "cfg7": "/path/to/predictions/reconsplat_dl3dv_n150_fid_images_cfg7",
                },
            },
        },
    },
    "supplement": {
        "description": "RE10k extrapolation classifier-free-guidance (CFG) ablation, for the supplement.",
        "output_path": "outputs/reconsplat_fid_stats_supplement.json",
        "groups": {
            "extrapolation": {
                "dataset_name": "re10k-test-extrapolation",
                "compute_kid": False,
                "paths": {
                    "cfg1": "/path/to/predictions/reconsplat_re10k_extrapolation_fid_images_cfg1",
                    "cfg5": "/path/to/predictions/reconsplat_re10k_extrapolation_fid_images_cfg5",
                    "cfg7": "/path/to/predictions/reconsplat_re10k_extrapolation_fid_images_cfg7",
                },
            },
        },
    },
}


def _dataset_name_for(group: dict, method: str) -> str:
    if "dataset_name" in group:
        return group["dataset_name"]
    return group["dataset_name_template"].format(method=method)


def run_experiment(name: str, batch_size: int, num_workers: int) -> None:
    cfg = EXPERIMENTS[name]
    print(f"=== {name}: {cfg['description']} ===")

    output_dict = {}

    for group_name, group in cfg["groups"].items():
        flatten = group.get("flatten", False)
        if not flatten:
            output_dict.setdefault(group_name, {})

        for method, path in group["paths"].items():
            dataset_name = _dataset_name_for(group, method)

            fid_score = fid.compute_fid(
                fdir1=path,
                dataset_name=dataset_name,
                mode="clean",
                batch_size=batch_size,
                num_workers=num_workers,
                dataset_split="custom",
                verbose=True,
            )
            entry = {"FID": fid_score}

            if group["compute_kid"]:
                kid_score = fid.compute_kid(
                    fdir1=path,
                    dataset_name=dataset_name,
                    mode="clean",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    dataset_split="custom",
                    verbose=True,
                )
                entry["KID"] = kid_score

            if flatten:
                output_dict[method] = entry
            else:
                output_dict[group_name][method] = entry

            print(f"  [{group_name}/{method}] dataset={dataset_name} {entry}")

    output_path = cfg["output_path"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    print(f"[OK] wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "experiments",
        nargs="*",
        help=f"Which registered experiment(s) to (re)run (choices: {', '.join(EXPERIMENTS.keys())}). "
             "Omit to run all of them.",
    )
    parser.add_argument("--list", action="store_true", help="List available experiments and exit.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(f"{name}: {cfg['description']} -> {cfg['output_path']}")
        return

    unknown = [n for n in args.experiments if n not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiment(s): {unknown} (choices: {', '.join(EXPERIMENTS.keys())})")

    names = args.experiments or list(EXPERIMENTS.keys())
    for name in names:
        run_experiment(name, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
