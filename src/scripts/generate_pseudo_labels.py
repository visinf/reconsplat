# Script to generate dense depth pseudo-labels.
# Steps:
# 1. Run VGGT (feed-forward) to predict camera poses, per-pixel depth  and confidence. 
#    Traditional SfM/bundle-adjustment is not yet supported here.
# 2. Optionally, run Video Depth Anything (VDA) to generate scale-consistent dense depth maps.
# 3. If step 2 ran, estimate a global scale/shift to align VGGT's dense depth (weighted
#    by its per-pixel confidence) with dense multi-view depth estimated by VDA.
# 4. Save the (aligned, if step 2/3 ran) depth maps and the camera poses predicted by
#    VGGT. 
# 5. Once an entire chunk has been processed, save the new depth/camera data to disk.

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
from src.labels.vggt.utils.colmap import (
    convert_vggt_recon_to_colmap,
    rename_colmap_recons_and_rescale_camera
)

from src.labels.video_depth_anything.video_depth import VideoDepthAnything
from src.labels.video_depth_anything.util.dc_utils import save_video
from src.misc.ransac import align_dense_depth_maps_global
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

    if labels_cfg.pipeline.align_with_video_depth:
        # Load Video-Depth-Anything model
        encoder = labels_cfg.pipeline.video_depth.encoder
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        video_depth_anything = VideoDepthAnything(**model_configs[encoder])
        model_path = os.path.join(labels_cfg.pipeline.video_depth.model_dir, f'video_depth_anything_{encoder}.pth')
        video_depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(device).eval()

        # Video-Depth-Anything params
        target_fps = 25

    rescale_predictions_to_orginal_res = labels_cfg.pipeline.scale_predictions_to_orginal_res
    if labels_cfg.pipeline.processor.from_chunks:
        save_back_to_chunks = labels_cfg.pipeline.processor.save_back_to_chunks
        save_cameras_to_npz = labels_cfg.pipeline.processor.save_cameras_to_npz

        print("Loading images from chunks...")
        path_to_chunks = labels_cfg.pipeline.load_dir
        splits = ['train', 'test']

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

                if save_cameras_to_npz:
                    # Initialize chunk list to save cameras data.
                    chunk_cameras = {}

                for scene_idx in range(len(chunk)):
                    scene_id = chunk[scene_idx]["key"]
                    print(f"Processing scene: {scene_id}")

                    raw_images = chunk[scene_idx]["images"]
                    pil_images = [Image.open(BytesIO(raw_images[t].numpy().tobytes())) for t in range(len(raw_images))]

                    if labels_cfg.pipeline.align_with_video_depth:
                        # Run Video-Depth-Anything to predict consistent affine-invariant depth for all the frames.
                        print(f"Predicting Video Depth...")
                        np_images = [np.array(pil_image) for pil_image in pil_images]
                        np_images = np.stack(np_images, axis=0)
                        print(f'Original video shape: {np_images.shape}')
                        depths, fps = video_depth_anything.infer_video_depth(np_images, target_fps)
                        print(f'Video depth shape: {depths.shape}')

                    if labels_cfg.pipeline.structure.feed_forward:
                        # Run feed-forward prediction of geometry with VGGT
                        # NOTE: SfM with BA is not yet supported by this script.
                        images, original_coords = load_and_preprocess_images(
                            pil_images, 
                            target_shape=img_load_resolution, 
                            patch_size=patch_size, 
                            mode="crop", 
                            return_coords=True
                        )
                        images = images.to(device)
                        
                        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(
                            model, 
                            images, 
                            dtype=dtype, 
                            resolution=vggt_fixed_resolution
                        )
                        print(f'VGGT depth map shape: {depth_map.shape}')
                        avg_depth_conf = np.average(depth_conf)
                        print(f'[INFO] avg. depth_conf = {avg_depth_conf}')
                        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

                        # Fallback so save_depth_as_npz always has a usable depth map, even when
                        # align_with_video_depth is off (in which case there's no alignment to
                        # override this with the aligned dense depth map below).
                        chunk[scene_idx]["dense_depth_map"] = depth_map.squeeze(-1)

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

                        if save_back_to_chunks:
                            # TODO: deprecate this... not recommended! Overwrite COLMAP cameras on the original chunks
                            chunk[scene_idx]["cameras"] = cameras
                        
                        if save_cameras_to_npz:
                            chunk_cameras[scene_id] = cameras
                        
                        # Save COLMAP reconstruction (intermediate results)
                        if labels_cfg.pipeline.structure.save_images:
                            print(f"Saving input VGGT images to {output_dir}/images")
                            image_dir = os.path.join(output_dir, "images")
                            os.makedirs(image_dir, exist_ok=True)
                            input_images = images.cpu()
                            input_images = [to_pil_image(input_images[t]) for t in range(input_images.shape[0])]
                            for t, input_image in enumerate(input_images):
                                input_image.save(f"{image_dir}/frame_{t:06d}.png")

                        if labels_cfg.pipeline.structure.save_intermediate_sparse:
                            # Create folder to save intermediate and final results.
                            output_dir = os.path.join(labels_cfg.pipeline.out_dir, split, 'structure', scene_id)
                            os.makedirs(output_dir, exist_ok=True)
                            sparse_reconstruction_dir = os.path.join(output_dir, "sparse")
                            os.makedirs(sparse_reconstruction_dir, exist_ok=True)

                            # Convert reconstruction to COLMAP format
                            reconstruction = convert_vggt_recon_to_colmap(
                                images, points_3d, depth_conf, conf_thres_value, max_points_for_colmap, extrinsic, intrinsic, vggt_fixed_resolution
                            )
                            reconstruction_resolution = vggt_fixed_resolution
                            
                            reconstruction = rename_colmap_recons_and_rescale_camera(
                                reconstruction, 
                                original_coords.cpu().numpy(), 
                                img_size=reconstruction_resolution, 
                                shift_point2d_to_original_res=rescale_predictions_to_orginal_res, 
                                rescale_camera_intr=rescale_predictions_to_orginal_res,
                                shared_camera=False
                            )
                            print(f"Saving reconstruction to {sparse_reconstruction_dir}")
                            reconstruction.write(sparse_reconstruction_dir)
                            reconstruction.export_PLY(os.path.join(sparse_reconstruction_dir, "points.ply"))

                        # TODO: Add support to scale depth maps to original resolution if scale_predictions_to_orginal_res is True.
                        if labels_cfg.pipeline.structure.save_intermediate_dense:
                            # Create folder to save intermediate results.
                            output_dir = os.path.join(labels_cfg.pipeline.out_dir, split, 'structure', scene_id)
                            os.makedirs(output_dir, exist_ok=True)
                            dense_reconstruction_dir = os.path.join(output_dir, "dense")
                            os.makedirs(dense_reconstruction_dir, exist_ok=True)

                            print(f"Saving dense (intermediate) pseudo-labels to {dense_reconstruction_dir}")
                            np.save(os.path.join(dense_reconstruction_dir, "vggt_depth_map.npy"), depth_map)
                            np.save(os.path.join(dense_reconstruction_dir, "vggt_depth_conf.npy"), depth_conf)

                        # Align dense depth maps from Video Depth Anything with VGGT results
                        if labels_cfg.pipeline.align_with_video_depth:
                            # Check whether video depth prediction and img_load_resolution have the same aspect ratio
                            # NOTE: Always true if the input video is rescaled to img_load_resolution before Video Depth Anything inference.
                            depth_ratio = depths.shape[1] / depths.shape[2]
                            img_load_ratio = img_load_resolution[0] / img_load_resolution[1]

                            if depth_ratio != img_load_ratio:
                                # save aspect ration mismatch error, but continue processing...
                                errors[scene_id] = f"Aspect ratio mismatch: depth {depth_ratio} vs img {img_load_ratio}"
   
                            depth_vda = rescale_and_crop_video_depth(torch.from_numpy(depths), img_load_resolution, vggt_fixed_resolution).numpy()
                            print(f"Video Depth Anything depths shape, after rescaling and crop: {depth_vda.shape}")
                            depth_map = np.squeeze(depth_map, axis=-1)

                            if labels_cfg.pipeline.structure.save_video_depth:
                                depth_vis_path = os.path.join(output_dir, 'video_depth_inferno.mp4')
                                save_video(depth_vda, depth_vis_path, fps=25, is_depths=True, grayscale=False)
                                vgg_depth_vis_path = os.path.join(output_dir, 'vggt_depth_inferno.mp4')
                                save_video(depth_map, vgg_depth_vis_path, fps=25, is_depths=True, grayscale=False, reverse_colormap=True)

                            image_ids = sorted([f'frame_{idx:06d}.png' for idx in range(images.shape[0])])
                            sfm_depth_dict, sfm_confidence_dict = dict(), dict()
                            disp_dict = dict()

                            for idx in range(len(image_ids)):
                                image_id = image_ids[idx]
                                sfm_depth_dict[image_id] = depth_map[idx]
                                sfm_confidence_dict[image_id] = depth_conf[idx]
                                disp_dict[image_id] = depth_vda[idx]

                            depth_dict = align_dense_depth_maps_global(
                                sfm_depth_dict,
                                sfm_confidence_dict,
                                disp_dict
                            )

                            depth_dict = dict(sorted(depth_dict.items()))
                            dense_depth_map = [np.array(depth_dict[image_id], dtype=depths.dtype) for image_id in depth_dict]
                            dense_depth_map = np.stack(dense_depth_map, axis=0)
                            chunk[scene_idx]["dense_depth_map"] = dense_depth_map
                            
                            if labels_cfg.pipeline.structure.save_intermediate_dense:
                                np.save(os.path.join(dense_reconstruction_dir, "aligned_dense_depth_map.npy"), dense_depth_map)

                            if labels_cfg.pipeline.save_aligned_depth_video:
                                dense_depth_vis_path = os.path.join(output_dir, 'dense_aligned_depth_video.mp4')
                                save_video(dense_depth_map, dense_depth_vis_path, fps=25, is_depths=True, grayscale=False, reverse_colormap=True)

                            if labels_cfg.pipeline.save_recon_after_align:  
                                dense_points_3d = unproject_depth_map_to_point_map(
                                    dense_depth_map[..., None], 
                                    extrinsic, 
                                    intrinsic
                                )

                                # assign minimum valid threshold to dense predictions.
                                B, _, H, W = images.shape
                                dense_depth_conf = np.ones((B, H, W)) * conf_thres_value
                                dense_recon = convert_vggt_recon_to_colmap(
                                    images, 
                                    dense_points_3d,
                                    dense_depth_conf,
                                    conf_thres_value,
                                    max_points_for_colmap=max_points_for_colmap * 5, # densify for pcd visualization
                                    extrinsic=extrinsic,     
                                    intrinsic=intrinsic,
                                    vggt_res=vggt_fixed_resolution
                                )

                                print(f"Saving aligned / dense reconstruction to {dense_reconstruction_dir}")
                                dense_recon.write(dense_reconstruction_dir)
                                dense_recon.export_PLY(os.path.join(dense_reconstruction_dir, "points.ply"))  
                
                # Get chunk.torch path and save updated chunk
                chunk_id = chunk_path.split('/')[-1].replace('.torch', '')
                
                depth_chunk_path = os.path.join(depths_out_dir, f"{chunk_id}.npz")
                camera_chunk_path = os.path.join(cameras_out_dir, f"{chunk_id}.npz")

                if labels_cfg.pipeline.processor.save_depth_as_npz:
                    # Save chunk with dense depth maps - only save updated fields, others can be retrieved from the original chunk.
                    print(f"Saving chunk {chunk_id} with updated meta and dense depth maps")
                    save_dict = {}
                    for scene_idx, scene in enumerate(chunk):
                        scene_id = scene["key"]
                        save_dict[scene_id] = scene['dense_depth_map']

                    np.savez_compressed(depth_chunk_path, **save_dict)

                if save_back_to_chunks:
                    chunk_path = os.path.join(labels_cfg.pipeline.out_dir, split, f'{chunk_id}.torch')
                    # otherwise save with torch.save
                    torch.save(chunk, chunk_path)
            
                if save_cameras_to_npz:
                    np.savez_compressed(camera_chunk_path, **chunk_cameras)

                if labels_cfg.pipeline.processor.save_progress:
                    # Save progress, which is useful for resuming the process.
                    # I guess this caching method can be improved a lot.
                    with open(progress_path, "a", buffering=1) as f:
                        f.write(f"{chunk_id}\n")
                    with open(errors_path, "w", encoding="utf-8") as f:
                        json.dump(errors, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_pseudo_labels()