import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from met3r import MEt3R


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path):
    """
    Sort paths naturally:
    frame_2.png before frame_10.png
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]


def load_image(path: Path, img_size: int, device: torch.device) -> torch.Tensor:
    """
    Loads an RGB image and returns tensor in [-1, 1],
    shape: (3, H, W)
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    if img_size is not None:
        img = img.resize((img_size, img_size), Image.BICUBIC)

    x = TF.to_tensor(img)          # [0, 1], shape (3, H, W)
    x = x * 2.0 - 1.0              # [-1, 1]
    return x.to(device)


@torch.no_grad()
def compute_scene_score(
    metric,
    image_paths,
    img_size: int,
    batch_size: int,
    device: torch.device,
):
    """
    Computes MEt3R between subsequent sorted views:
    image_0 vs image_1,
    image_1 vs image_2,
    ...
    """
    pair_scores = []

    num_pairs = len(image_paths) - 1

    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)

        batch_pairs = []

        for i in range(start, end):
            img_a = load_image(image_paths[i], img_size, device)
            img_b = load_image(image_paths[i + 1], img_size, device)

            pair = torch.stack([img_a, img_b], dim=0)  # (2, 3, H, W)
            batch_pairs.append(pair)

        inputs = torch.stack(batch_pairs, dim=0)  # (B, 2, 3, H, W)

        score, *_ = metric(
            images=inputs,
            return_overlap_mask=False,
            return_score_map=False,
            return_projections=False,
        )

        pair_scores.append(score.detach().float().cpu())

    pair_scores = torch.cat(pair_scores, dim=0)

    return {
        "mean": pair_scores.mean().item(),
        "std": pair_scores.std(unbiased=False).item(),
        "num_pairs": len(pair_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="met3r_scores.json")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--backbone", type=str, default="mast3r")
    parser.add_argument("--feature_backbone", type=str, default="dino16")
    parser.add_argument("--feature_backbone_weights", type=str, default="mhamilton723/FeatUp")
    parser.add_argument("--upsampler", type=str, default="featup")
    parser.add_argument("--distance", type=str, default="cosine")
    parser.add_argument("--use_norm", action="store_true", default=True)

    args = parser.parse_args()

    root_folder = Path(args.root_folder)
    output_json = Path(args.output_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric = MEt3R(
        img_size=args.img_size,
        use_norm=args.use_norm,
        backbone=args.backbone,
        feature_backbone=args.feature_backbone,
        feature_backbone_weights=args.feature_backbone_weights,
        upsampler=args.upsampler,
        distance=args.distance,
        freeze=True,
    ).to(device)

    metric.eval()

    scene_rows = []

    scene_folders = sorted(
        [p for p in root_folder.iterdir() if p.is_dir()],
        key=natural_key,
    )

    for scene_folder in scene_folders:
        color_folder = scene_folder / "color"

        if not color_folder.exists():
            print(f"[skip] {scene_folder.name}: no color folder")
            continue

        image_paths = sorted(
            [
                p for p in color_folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
            ],
            key=natural_key,
        )

        if len(image_paths) < 2:
            print(f"[skip] {scene_folder.name}: fewer than 2 images")
            continue

        print(f"[scene] {scene_folder.name}: {len(image_paths)} images")

        result = compute_scene_score(
            metric=metric,
            image_paths=image_paths,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=device,
        )

        scene_rows.append({
            "scene_id": scene_folder.name,
            "num_images": len(image_paths),
            "num_pairs": result["num_pairs"],
            "mean_met3r": result["mean"],
            "std_met3r": result["std"],
        })

        print(
            f"  mean={result['mean']:.6f}, "
            f"std={result['std']:.6f}, "
            f"pairs={result['num_pairs']}"
        )

        torch.cuda.empty_cache()

    if len(scene_rows) == 0:
        raise RuntimeError("No valid scenes found.")

    # Macro average: each scene has equal weight
    dataset_macro_mean = sum(row["mean_met3r"] for row in scene_rows) / len(scene_rows)

    # Weighted average: each image pair has equal weight
    total_pairs = sum(row["num_pairs"] for row in scene_rows)
    dataset_weighted_mean = (
        sum(row["mean_met3r"] * row["num_pairs"] for row in scene_rows)
        / total_pairs
    )

    results = {
        "num_scenes": len(scene_rows),
        "total_pairs": total_pairs,
        "dataset_macro_mean": dataset_macro_mean,
        "dataset_weighted_mean": dataset_weighted_mean,
        "scenes": scene_rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n==============================")
    print(f"Saved scores to: {output_json}")
    print(f"Number of scenes: {len(scene_rows)}")
    print(f"Dataset macro mean:    {dataset_macro_mean:.6f}")
    print(f"Dataset weighted mean: {dataset_weighted_mean:.6f}")
    print("==============================")

if __name__ == "__main__":
    main()