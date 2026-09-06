import numpy as np
import torch
import torch.nn.functional as F
from .helper import create_pixel_coordinate_grid, randomly_limit_trues
from ..dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
import copy

def convert_vggt_recon_to_colmap(
        images, 
        points_3d,
        depth_conf,
        conf_thres_value,
        max_points_for_colmap,
        extrinsic, 
        intrinsic, 
        vggt_res, 
        shared_camera=False, 
        camera_type="PINHOLE"):
    
    vggt_height, vggt_width = vggt_res
    image_size = np.array([vggt_width, vggt_height])
    num_frames, height, width, _ = points_3d.shape
    points_rgb = F.interpolate(
        images, size=(vggt_height, vggt_width), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)
    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    conf_mask = depth_conf >= conf_thres_value
    # at most writing <max_points_for_colmap> 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )
    return reconstruction

def convert_dense_dict_to_colmap(
    unproj_dense_points3D: dict[str, np.ndarray],
    extrinsic: np.ndarray,        # (F,4,4) or list[4x4] – one per frame
    intrinsic: np.ndarray,        # (F,...)  – same order as `extrinsic`
    image_size: tuple[int, int],  # (height, width)
    shared_camera=False,
    camera_type="PINHOLE",
):
    """
    Convert the {img_path: [points_world, rgb]} dictionary returned by
    `align_dense_depth_maps` into a pycolmap.Reconstruction.
    """
    pts3d_lst, rgb_lst, xyf_lst = [], [], []
    img_name_to_frame = {name: i for i, name in enumerate(unproj_dense_points3D)}

    for frame_idx, (img_name, packed) in enumerate(unproj_dense_points3D.items()):
        xyz, rgb = packed                 # xyz.shape == (n_i,3), rgb == (n_i,3)
        n_i = xyz.shape[0]

        # ----------- build dummy (x,y,f) -----------------
        xyf = np.zeros((n_i, 3), dtype=np.float32)
        xyf[:, 2] = frame_idx             # store frame index in the 3rd column

        # ----------- collect -----------------------------
        pts3d_lst.append(xyz)
        rgb_lst.append((rgb * 255).astype(np.uint8))
        xyf_lst.append(xyf)

    # --------------- flatten -----------------------------
    points_3d  = np.concatenate(pts3d_lst, axis=0)
    points_rgb = np.concatenate(rgb_lst,  axis=0)
    points_xyf = np.concatenate(xyf_lst,  axis=0)

    # --------------- hand‑off to your helper -------------
    recon = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        np.array(image_size),             # (H,W)
        shared_camera=shared_camera,
        camera_type=camera_type,
    )
    return recon


def rename_colmap_recons_and_rescale_camera(
    reconstruction, original_coords, img_size, image_paths=None, shift_point2d_to_original_res=False, rescale_camera_intr=False, shared_camera=False
):
    if shift_point2d_to_original_res:
        assert rescale_camera is True

    rescale_camera = rescale_camera_intr
    is_rectangular = original_coords.shape[1] == 6

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        if image_paths is not None:
            pyimage.name = image_paths[pyimageid - 1]
        else:
            pyimage.name = f'frame_{pyimageid - 1:06d}.png'
        
        if rescale_camera:
            if is_rectangular:
                (x_off, y_off, scale_x, scale_y, real_w, real_h) = original_coords[pyimageid - 1]
                resize_ratio_x = 1.0 / scale_x
                resize_ratio_y = 1.0 / scale_y
            else:
                real_w, real_h = original_coords[pyimageid - 1, -2:]
                resize_ratio_x = resize_ratio_y = max(real_w, real_h) / img_size
                x_off, y_off = original_coords[pyimageid - 1, :2]

            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            # rescale focal length
            pred_params[0] *= resize_ratio_x
            pred_params[1] *= resize_ratio_y

            # centre of projection in original pixels
            pred_params[-2:] = np.array([real_w, real_h]) / 2.0

            pycamera.params = pred_params
            pycamera.width = int(real_w)
            pycamera.height = int(real_h)

            if shift_point2d_to_original_res:
                # Also shift the point2D to original resolution
                for point2D in pyimage.points2D:
                    point2D.xy[0] = (point2D.xy[0] - x_off) * resize_ratio_x
                    point2D.xy[1] = (point2D.xy[1] - y_off) * resize_ratio_y

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction
