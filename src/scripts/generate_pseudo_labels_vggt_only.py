import os
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import random
import hydra
import json
from omegaconf import DictConfig
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from src.labels.vggt.models.vggt import VGGT
from src.labels.vggt.utils.load_fn import load_and_preprocess_images
from src.labels.vggt.utils.geometry import unproject_depth_map_to_point_map
from src.labels.pseudo_label_utils import (
    get_dist_info,
    collect_processed_chunks,
    run_VGGT,
    rescale_and_crop_video_depth,
)

from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.global_cfg import set_cfg
    from src.labels.config import PseudoLabelsCfg

# TODO: Can we do BA on top on VGGT with non-square images?
# For now, let's get pseudo-labels from feed-forward VGGT.
@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="labels",
)
def generate_pseudo_labels(cfg_dict: DictConfig):
    rank, world_size = get_dist_info()
    labels_cfg = load_typed_config(
        cfg_dict, 
        PseudoLabelsCfg
    )
    set_cfg(cfg_dict)
    
    seed = labels_cfg.seed
    if seed is not None:
        # Set seed for reproducibility
        # Add rank to seed, so each GPU gets independent, reproducible stream.
        seed_for_this_rank = seed + rank
        np.random.seed(seed_for_this_rank)
        torch.manual_seed(seed_for_this_rank)
        random.seed(seed_for_this_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_for_this_rank)
            torch.cuda.manual_seed_all(seed_for_this_rank)  # for multi-GPU
        print(f"Setting seed as: {seed_for_this_rank} (base {seed} + rank {rank})")
    
    # Set device and dtype
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        major_cap, _ = torch.cuda.get_device_capability(device)
    else:
        device = "cpu"
        major_cap = 0

    dtype = torch.bfloat16 if major_cap >= 8 else torch.float16

    print(f"Rank {rank}/{world_size} — device: {device}, dtype: {dtype}")
    print(f"Using labels configuration: {labels_cfg}")

    if labels_cfg.pipeline.structure.feed_forward:
        # Load VGGT for camera and depth estimation
        model = VGGT()
        if labels_cfg.pipeline.structure.load_from_url:
            print("Loading model from URL...")
            _URL = labels_cfg.pipeline.structure.model_path
            model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
            model.eval()
            model = model.to(device)
            print(f"Model loaded")
        else:
            print("Loading model from local path...")
            model.load_state_dict(torch.load(labels_cfg.pipeline.structure.model_path, map_location=device))
            model.eval()
            model = model.to(device)
            print(f"Model loaded from {labels_cfg.pipeline.structure.model_path}")

        # VGGT params
        img_load_resolution = labels_cfg.pipeline.processor.image_load_resolution
        vggt_fixed_resolution = labels_cfg.pipeline.structure.vggt_fixed_resolution
        patch_size = labels_cfg.pipeline.structure.patch_size
        conf_thres_value = labels_cfg.pipeline.structure.conf_thres_value
        max_points_for_colmap = labels_cfg.pipeline.structure.max_points_for_colmap

    rescale_predictions_to_orginal_res = labels_cfg.pipeline.scale_predictions_to_orginal_res
    if labels_cfg.pipeline.processor.from_chunks:
        save_back_to_chunks = labels_cfg.pipeline.processor.save_back_to_chunks
        save_cameras_to_npz = labels_cfg.pipeline.processor.save_cameras_to_npz

        print("Loading images from chunks...")
        path_to_chunks = labels_cfg.pipeline.load_dir
        # Process only train split - we already have the test split.
        splits = ['train']

        for split in splits:
            print(f"Processing split: {split}")
            split_path = os.path.join(path_to_chunks, split)
            all_chunk_paths = sorted([os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.torch')])
            print(f"Found {len(all_chunk_paths)} chunks in {split_path}")
            chunk_paths = all_chunk_paths[rank::world_size]  # Split chunks across GPUs
            print(f"Rank {rank} processing {len(chunk_paths)} chunks from split {split}")

            split_out_dir = os.path.join(labels_cfg.pipeline.out_dir, split)
            os.makedirs(split_out_dir, exist_ok=True)
            cameras_out_dir = os.path.join(split_out_dir, "cameras")
            os.makedirs(cameras_out_dir, exist_ok=True)
            depths_out_dir = os.path.join(split_out_dir, "depths")
            os.makedirs(depths_out_dir, exist_ok=True)
            confs_out_dir = os.path.join(split_out_dir, "confs")
            os.makedirs(confs_out_dir, exist_ok=True)

            progress_dir = os.path.join(split_out_dir, labels_cfg.pipeline.processor.progress_dir)
            errors_dir = os.path.join(split_out_dir, labels_cfg.pipeline.processor.errors_dir)
            errors = {}

            # NOTE: once resumed, the skip_cache_file will contain the chunks already processed (by all ranks).
            #       individual files for each rank are instead initialized from scratch.
            if labels_cfg.pipeline.processor.skip_already_processed_chunks:
                # Collect already processed chunks from all ranks
                processed_chunks_global = collect_processed_chunks(rank, progress_dir, cache_file=labels_cfg.pipeline.processor.skip_cache_file)
                chunk_paths = [p for p in chunk_paths if os.path.basename(p).replace(".torch", "") not in processed_chunks_global]
                print(f"Rank {rank}: {len(processed_chunks_global)} chunks already done")

            if labels_cfg.pipeline.processor.save_progress:
                os.makedirs(progress_dir, exist_ok=True)
                os.makedirs(errors_dir, exist_ok=True)
                progress_path = os.path.join(progress_dir, f"processed_chunks_rank{rank}.txt")
                errors_path = os.path.join(errors_dir, f"errors_rank{rank}.json")

            for chunk_path in tqdm(chunk_paths, desc=f"Rank {rank}: processing {split} chunks"):
                chunk = torch.load(chunk_path, weights_only=True)

                chunk_cameras = {}
                chunk_depths = {}
                chunk_conf = {}

                for scene_idx in range(len(chunk)):
                    scene_id = chunk[scene_idx]["key"]
                    print(f"Processing scene: {scene_id}")

                    raw_images = chunk[scene_idx]["images"]
                    pil_images = [Image.open(BytesIO(raw_images[t].numpy().tobytes())) for t in range(len(raw_images))]

                    # Run feed-forward prediction of geometry with VGGT
                    # NOTE: SfM with BA is not yet supported by this script.
                    try:
                        images, original_coords = load_and_preprocess_images(
                            pil_images, 
                            target_shape=img_load_resolution, 
                            patch_size=patch_size, 
                            mode="crop", 
                            return_coords=True
                        )
                        images = images.to(device)
                    except Exception as e:
                        print(f"An error in opening images for {scene_id} occured:", e)
                        print(f"skipping {scene_id} to continue...")
                        continue
                    
                    if images.shape[0] > 450:
                        print(f"Skipping scene {scene_id}: scene is too big to be processed by VGGT.")
                        continue
                    
                    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(
                        model, 
                        images, 
                        dtype=dtype, 
                        resolution=vggt_fixed_resolution
                    )
                    print(f'VGGT depth map shape: {depth_map.shape}')

                    # Save cameras back to original chunks or into a chunk-wise json format.
                    intrinsic_t = torch.from_numpy(intrinsic)
                    extrinsic_t = torch.from_numpy(extrinsic)
                    zeros = torch.zeros((extrinsic_t.shape[0], 2), dtype=extrinsic_t.dtype)
                    intr = torch.stack(
                        (
                            intrinsic_t[:, 0, 0], 
                            intrinsic_t[:, 1, 1], 
                            intrinsic_t[:, 0, 2], 
                            intrinsic_t[:, 1, 2],
                            zeros[:, 0],
                            zeros[:, 1]
                        ),
                        dim=1
                    )
                    extr = extrinsic_t.reshape(extrinsic_t.shape[0], -1)
                    cameras = torch.cat((intr, extr), dim=1)

                    chunk_cameras[scene_id] = cameras
                    chunk_depths[scene_id] = depth_map.squeeze(-1)
                    chunk_conf[scene_id] = depth_conf

                # Get chunk.torch path and save updated chunk
                chunk_id = chunk_path.split('/')[-1].replace('.torch', '')
                
                depth_chunk_path = os.path.join(depths_out_dir, f"{chunk_id}.npz")
                camera_chunk_path = os.path.join(cameras_out_dir, f"{chunk_id}.npz")
                conf_chunk_path = os.path.join(confs_out_dir, f"{chunk_id}.npz")

                # Save depth chunk.
                np.savez_compressed(depth_chunk_path, **chunk_depths)

                # Save camera chunk.
                np.savez_compressed(camera_chunk_path, **chunk_cameras)

                # Save conf chunk.
                np.savez_compressed(conf_chunk_path, **chunk_conf)

                if labels_cfg.pipeline.processor.save_progress:
                    # Save progress, which is useful for resuming the process.
                    # I guess this caching method can be improved a lot.
                    with open(progress_path, "a", buffering=1) as f:
                        f.write(f"{chunk_id}\n")
                    with open(errors_path, "w", encoding="utf-8") as f:
                        json.dump(errors, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_pseudo_labels()