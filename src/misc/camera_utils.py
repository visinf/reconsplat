import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, repeat
import numpy as np

def absolute_to_relative_camera(
    tform: Float[Tensor, "batch v 4 4"],
    index: int
):
    _, v, *_ = tform.shape
    ref_tform = tform[:, index:index+1, ...]
    ref_tform = ref_tform.expand(-1, v, -1, -1) 

    # tform = tform.inverse() @ ref_tform
    tform = torch.linalg.inv(ref_tform) @ tform
    
    # new_tform = torch.zeros_like(ref_tform)
    # new_tform[..., :3, :3] = ref_tform[..., :3, :3].transpose(-1, -2)
    # new_tform[..., :3, 3] = -ref_tform[..., :3, 3]
    
    # tform = new_tform @ tform
    
    return tform

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def convert_poses(
    poses: Float[Tensor, "batch 18"],
) -> tuple[
    Float[Tensor, "batch 4 4"],  # extrinsics
    Float[Tensor, "batch 3 3"],  # intrinsics
]:
    '''
    Convert poses from chunking format (RE10k, DL3DV10k,...) to Bx4x4 extrinsics and Bx3x3 intrinsics.
    '''
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics

def denormalize_K(K_norm, width, height):
    K_pix = K_norm.clone()
    K_pix[:, 0, 0] *= width
    K_pix[:, 1, 1] *= height
    K_pix[:, 0, 2] *= width
    K_pix[:, 1, 2] *= height
    return K_pix

def normalize_K(K_pix, width, height):
    K_norm = K_pix.clone()
    K_norm[:, 0, 0] /= width
    K_norm[:, 1, 1] /= height
    K_norm[:, 0, 2] /= width
    K_norm[:, 1, 2] /= height
    return K_norm

def meshgrid_xy(batch_size: int, height: int, width: int, device=None):
    """
    Returns homogeneous pixel coordinates of shape:
    (B, 3, H, W), where coords[b,:,y,x] = [x, y, 1].
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    # (3, H, W)
    coords = torch.stack([xs, ys, ones], dim=0)
    # (B, 3, H, W)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return coords


def backproject_depth(
    depth: torch.Tensor,
    K: torch.Tensor,
    T_world_cam: torch.Tensor,
):
    """
    Back-project pixels + depth to 3D world coordinates.

    Args:
        depth: (B, 1, H, W) depth in the *camera* coordinate frame (z forward).
        K: (B, 3, 3) intrinsics
        T_world_cam: (B, 4, 4) camera-to-world transform

    Returns:
        X_world: (B, 3, H, W) 3D points in world coordinates.
    """
    B, _, H, W = depth.shape
    device = depth.device

    # homogeneous pixel coords (B, 3, H, W)
    pix_coords = meshgrid_xy(B, H, W, device=device)  # [x, y, 1]

    # Inverse intrinsics: (B, 3, 3)
    K_inv = torch.inverse(K)

    # cam_coords = depth * K^{-1} * [x, y, 1]^T
    # -> (B, 3, H, W)
    cam_coords = K_inv @ pix_coords.view(B, 3, -1)
    cam_coords = cam_coords.view(B, 3, H, W)
    cam_coords = cam_coords * depth  # multiply z along each ray

    # Convert to homogeneous: (B, 4, H, W)
    ones = torch.ones_like(cam_coords[:, :1])
    cam_coords_h = torch.cat([cam_coords, ones], dim=1)

    # world_coords_h = T_world_cam @ cam_coords_h
    # T_world_cam: (B, 4, 4), cam_coords_h: (B, 4, H*W)
    world_coords_h = T_world_cam @ cam_coords_h.view(B, 4, -1)
    world_coords_h = world_coords_h.view(B, 4, H, W)

    # strip homogeneous
    X_world = world_coords_h[:, :3] / (world_coords_h[:, 3:4] + 1e-8)
    return X_world

def project_world_to_cam(
    X_world: torch.Tensor,
    K: torch.Tensor,
    T_world_cam: torch.Tensor,
):
    """
    Project world-space points to pixel coords and depths in camera j.

    Args:
        X_world: (B, 3, H, W)
        K: (B, 3, 3)
        T_world_cam: (B, 4, 4) camera-to-world. We'll invert to get T_cam_world.

    Returns:
        pix: (B, 2, H, W) pixel coords (x, y) in image j.
        depth: (B, 1, H, W) depth along camera z-axis.
    """
    B, _, H, W = X_world.shape

    # T_cam_world = T_world_cam^{-1}
    T_cam_world = torch.inverse(T_world_cam)

    # to homogeneous (B, 4, H, W)
    ones = torch.ones_like(X_world[:, :1])
    X_world_h = torch.cat([X_world, ones], dim=1)

    # camera coords: (B, 4, H*W)
    X_cam_h = T_cam_world @ X_world_h.view(B, 4, -1)
    X_cam_h = X_cam_h.view(B, 4, H, W)

    # (B, 3, H, W)
    X_cam = X_cam_h[:, :3]
    z = X_cam[:, 2:3]  # depth in this camera

    # project: (B, 3, H, W) -> (B, 3, H*W)
    pix_h = K @ X_cam.view(B, 3, -1)
    pix_h = pix_h.view(B, 3, H, W)
    # pixel coords
    x = pix_h[:, 0] / (pix_h[:, 2] + 1e-8)
    y = pix_h[:, 1] / (pix_h[:, 2] + 1e-8)
    pix = torch.stack([x, y], dim=1)  # (B, 2, H, W)

    return pix, z

def warp_view_i_to_j(
    image_i: torch.Tensor,   # (B, 3, H, W)
    depth_i: torch.Tensor,   # (B, 1, H, W)
    K_i: torch.Tensor,       # (B, 3, 3)
    T_world_cam_i: torch.Tensor,  # (B, 4, 4)
    image_j: torch.Tensor,   # (B, 3, H, W)
    depth_j: torch.Tensor,   # (B, 1, H, W)
    K_j: torch.Tensor,       # (B, 3, 3)
    T_world_cam_j: torch.Tensor,  # (B, 4, 4)
):
    """
    Warp from view i to j using depth_i, and also sample depth_j at projected coords.

    Returns:
        C_i: (B, 3, H, W) original color (same as image_i)
        C_i2j: (B, 3, H, W) color from view j reprojected to i
        Z_i2j: (B, 1, H, W) geometric depth (from i's 3D point) in j
        D_j_warped: (B, 1, H, W) depth_j sampled at projected pixels
        valid_mask: (B, 1, H, W) mask where projection lies inside j and z>0
    """
    B, _, H, W = image_i.shape
    device = image_i.device

    # 1) back-project pixels from i to world
    X_world = backproject_depth(depth_i, K_i, T_world_cam_i)  # (B, 3, H, W)

    # 2) project world → camera j
    pix_j, z_i2j = project_world_to_cam(X_world, K_j, T_world_cam_j)
    # pix_j: (B, 2, H, W) in pixel coords

    # 3) normalize pixel coords to [-1, 1] for grid_sample
    #    x_n = 2 * (x / (W-1)) - 1
    #    y_n = 2 * (y / (H-1)) - 1
    x = pix_j[:, 0]
    y = pix_j[:, 1]
    x_norm = 2.0 * (x / (W - 1)) - 1.0
    y_norm = 2.0 * (y / (H - 1)) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, H, W, 2)

    # 4) sample image_j and depth_j at that grid
    C_i2j = F.grid_sample(
        image_j,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    D_j_warped = F.grid_sample(
        depth_j,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # 5) valid mask: inside image bounds AND positive depth in j
    valid_x = (x_norm > -1.0) & (x_norm < 1.0)
    valid_y = (y_norm > -1.0) & (y_norm < 1.0)
    valid_z = (z_i2j > 0.0).squeeze(1)
    valid = (valid_x & valid_y & valid_z).unsqueeze(1).float()  # (B, 1, H, W)

    return image_i, C_i2j, z_i2j, D_j_warped, valid