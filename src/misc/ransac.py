import numpy as np
import pycolmap
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor  
from collections import defaultdict
from tqdm import tqdm
import random

# from VGGSfM (https://github.com/facebookresearch/vggsfm/blob/e1d9d2eb2b3575525792206fb94b2c749c58dc50/vggsfm/runners/runner.py#L744)
def extract_sparse_depth_and_point_from_reconstruction(predictions):
    """
    Extracts sparse depth and 3D points from the reconstruction.

    Args:
        predictions (dict): Contains reconstruction data with a 'reconstruction' key.

    Returns:
        dict: Updated predictions with 'sparse_depth' and 'sparse_point' keys.
    """
    reconstruction = predictions["reconstruction"]
    sparse_depth = defaultdict(list)
    sparse_point = defaultdict(list)
    # Extract sparse depths from SfM points
    for point3D_idx in reconstruction.points3D:
        pt3D = reconstruction.points3D[point3D_idx]
        for track_element in pt3D.track.elements:
            pyimg = reconstruction.images[track_element.image_id]
            pycam = reconstruction.cameras[pyimg.camera_id]
            img_name = pyimg.name
            projection = pyimg.cam_from_world * pt3D.xyz
            depth = projection[-1]
            # NOTE: uv here cooresponds to the (x, y)
            # at the original image coordinate
            # instead of the padded&resized one
            uv = pycam.img_from_cam(projection)
            sparse_depth[img_name].append(np.append(uv, depth))
            sparse_point[img_name].append(np.append(pt3D.xyz, point3D_idx))
    predictions["sparse_depth"] = sparse_depth
    predictions["sparse_point"] = sparse_point
    return predictions

# adapted from VGGSfM (https://github.com/facebookresearch/vggsfm/blob/e1d9d2eb2b3575525792206fb94b2c749c58dc50/vggsfm/utils/utils.py#L635)
# NOTE: performs frame-wise alignment (i.e., scale and shift estimation) with RANSAC
def align_dense_depth_maps(
    reconstruction, # reconstruction from SfM
    sfm_depth, # depth predictions from SfM
    sfm_depth_confidence, # depth confidence from SfM
    disp_dict   # depth predictions from a dense depth model
):
    # Define disparity and depth limits
    disparity_max = 10000
    disparity_min = 0.0001
    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    depth_dict = {}
    for img_basename in tqdm(
        sfm_depth, desc="Load monocular depth and Align"
    ):
        disp_map = disp_dict[img_basename]
    
        # Note that dense depth maps may have some invalid values such as sky
        # they are marked as 0, hence filter out 0 from the sampled depths
        sfm_depths = np.array(sfm_depth[img_basename])
        sfm_confs = np.array(sfm_depth_confidence[img_basename])
        positive_mask = disp_map > 0
        confidence_mask = sfm_confs > 1.0
        mask = positive_mask & confidence_mask

        sampled_disps = disp_map[mask]
        sfm_depths = sfm_depths[mask]

        sfm_depths = np.clip(sfm_depths, depth_min, depth_max)

        thres_ratio = 30
        target_disps = 1 / sfm_depths

        # RANSAC
        X = sampled_disps.reshape(-1, 1)
        y = target_disps
        ransac_thres = np.median(y) / thres_ratio

        if ransac_thres <= 0:
            raise ValueError("Ill-posed scene for depth alignment")

        ransac = RANSACRegressor(
            LinearRegression(),
            min_samples=2,
            residual_threshold=ransac_thres,
            max_trials=20000,
            loss="squared_error",
        )
        ransac.fit(X, y)
        scale = ransac.estimator_.coef_[0]
        shift = ransac.estimator_.intercept_
        # inlier_mask = ransac.inlier_mask_
        nonzero_mask = disp_map != 0
        # Rescale the disparity map
        disp_map[nonzero_mask] = disp_map[nonzero_mask] * scale + shift

        valid_depth_mask = (disp_map > 0) & (disp_map <= disparity_max)
        disp_map[~valid_depth_mask] = 0

        # Convert the disparity map to depth map
        depth_map = np.full(disp_map.shape, np.inf)
        depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.astype(np.float32)

        depth_dict[img_basename] = depth_map

    return depth_dict

# NOTE: performs global alignment with huber regressor
def sample_valid_pixels(disp, target_disp, conf_mask, max_pix=2000):
    m = conf_mask & (disp > 0) & np.isfinite(target_disp)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None, None
    if idx.size > max_pix:
        idx = np.random.choice(idx, max_pix, replace=False)
    return disp.ravel()[idx], target_disp.ravel()[idx]

def estimate_global_scale_shift(sfm_depth, disp_dict, conf_dict=None,
                                max_pix_per_img=2000):
    xs, ys = [], []
    for name in sfm_depth:
        disp = disp_dict[name].astype(np.float32)
        tgt_disp = 1. / np.clip(sfm_depth[name], 1e-4, 1e4).astype(np.float32)
        conf = conf_dict[name] if conf_dict else np.ones_like(disp, bool)
        conf_mask = conf >= 1.0
        # print("conf_mask: ", conf_mask.sum())
        x, y = sample_valid_pixels(disp, tgt_disp, conf_mask, max_pix_per_img)
        if x is not None:
            xs.append(x)
            ys.append(y)

    X = np.concatenate(xs)[:, None]          # shape (M,1)
    y = np.concatenate(ys)                   # shape (M,)

    huber = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True)
    huber.fit(X, y)

    alpha = float(huber.coef_[0])
    beta = float(huber.intercept_)
    return alpha, beta

def align_dense_depth_maps_global(
    sfm_depth, # depth predictions from SfM
    sfm_depth_confidence, # depth confidence from SfM
    disp_dict   # depth predictions from a dense depth model
):
    # debugging assertions
    assert isinstance(sfm_depth, dict), "sfm_depth should be a dictionary"
    assert isinstance(sfm_depth_confidence, dict), "sfm_depth_confidence should be a dictionary"
    assert isinstance(disp_dict, dict), "disp_dict should be a dictionary"
    # assert they are not empty
    assert len(sfm_depth) > 0, "sfm_depth is empty"
    assert len(sfm_depth_confidence) > 0, "sfm_depth_confidence is empty"
    assert len(disp_dict) > 0, "disp_dict is empty"

    # Estimate global scale and shift
    scale, shift = estimate_global_scale_shift(
        sfm_depth,
        disp_dict,
        sfm_depth_confidence
    )

    depth_dict = {}
    for name, disp in disp_dict.items():
        m = disp > 0
        disp[m]  = scale * disp[m] + shift
        depth_m  = np.empty_like(disp, dtype=np.float32)
        depth_m[m] = 1.0 / disp[m]
        depth_m[~m] = 0
        depth_dict[name] = depth_m
        
    return depth_dict