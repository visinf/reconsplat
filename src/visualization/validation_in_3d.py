import torch
from jaxtyping import Float, Shaped
from torch import Tensor

from ..model.decoder.cuda_splatting import render_cuda_orthographic
from ..model.types import Gaussians
from ..visualization.annotation import add_label
from ..visualization.drawing.cameras import draw_cameras
from .drawing.cameras import compute_equal_aabb_with_margin


def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results


def render_projections(
    gaussians: Gaussians,
    resolution: int,
    margin: float = 0.1,
    draw_label: bool = True,
    extra_label: str = "",
) -> Float[Tensor, "batch 3 3 height width"]:
    device = gaussians.means.device
    b, _, _ = gaussians.means.shape

    # Compute the minima and maxima of the scene.
    minima = gaussians.means.min(dim=1).values
    maxima = gaussians.means.max(dim=1).values
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(
        minima, maxima, margin=margin
    )

    projections = []
    for look_axis in range(3):
        right_axis = (look_axis + 1) % 3
        down_axis = (look_axis + 2) % 3

        # Define the extrinsics for rendering.
        extrinsics = torch.zeros((b, 4, 4), dtype=torch.float32, device=device)
        extrinsics[:, right_axis, 0] = 1
        extrinsics[:, down_axis, 1] = 1
        extrinsics[:, look_axis, 2] = 1
        extrinsics[:, right_axis, 3] = 0.5 * (
            scene_minima[:, right_axis] + scene_maxima[:, right_axis]
        )
        extrinsics[:, down_axis, 3] = 0.5 * (
            scene_minima[:, down_axis] + scene_maxima[:, down_axis]
        )
        extrinsics[:, look_axis, 3] = scene_minima[:, look_axis]
        extrinsics[:, 3, 3] = 1

        # Define the intrinsics for rendering.
        extents = scene_maxima - scene_minima
        far = extents[:, look_axis]
        near = torch.zeros_like(far)
        width = extents[:, right_axis]
        height = extents[:, down_axis]

        projection = render_cuda_orthographic(
            extrinsics,
            width,
            height,
            near,
            far,
            (resolution, resolution),
            torch.zeros((b, 3), dtype=torch.float32, device=device),
            gaussians.means,
            gaussians.covariances,
            gaussians.harmonics,
            gaussians.opacities,
            fov_degrees=10.0,
            gaussian_color_feature_sh_coefficients=gaussians.color_feature_harmonics,
            gaussian_geometry_features=gaussians.geometry_features
        )
        if draw_label:
            right_axis_name = "XYZ"[right_axis]
            down_axis_name = "XYZ"[down_axis]
            label = f"{right_axis_name}{down_axis_name} Projection {extra_label}"
            projection = torch.stack([add_label(x, label) for x in projection])

        projections.append(projection)

    return torch.stack(pad(projections), dim=1)


def render_cameras(batch: dict, resolution: int) -> Float[Tensor, "3 3 height width"]:
    # Define colors for context and target views.
    num_context_views = batch["context"]["extrinsics"].shape[1]
    num_target_views = batch["target"]["extrinsics"].shape[1]
    color = torch.ones(
        (num_target_views + num_context_views, 3),
        dtype=torch.float32,
        device=batch["target"]["extrinsics"].device,
    )
    color[num_context_views:, 1:] = 0

    return draw_cameras(
        resolution,
        torch.cat(
            (batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0])
        ),
        torch.cat(
            (batch["context"]["intrinsics"][0], batch["target"]["intrinsics"][0])
        ),
        color,
        torch.cat((batch["context"]["near"][0], batch["target"]["near"][0])),
        torch.cat((batch["context"]["far"][0], batch["target"]["far"][0])),
    )

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def compute_equal_aabb_with_margin(minima, maxima, margin: float = 0.1):
    """
    minima/maxima: (B, 3)
    Returns scene_minima, scene_maxima: (B, 3) with equalized extents + margin.
    """
    extents = maxima - minima  # (B,3)
    max_extent = extents.max(dim=-1, keepdim=True).values  # (B,1)
    center = 0.5 * (minima + maxima)

    half = 0.5 * max_extent * (1.0 + margin)
    scene_minima = center - half
    scene_maxima = center + half
    return scene_minima, scene_maxima


def _intrinsics_to_fxfycxcy(K: torch.Tensor):
    """
    K can be (B,V,3,3) or (B,3,3). Returns fx,fy,cx,cy broadcastable to (B,V,1,1).
    """
    if K.dim() == 3:
        K = K[:, None]  # (B,1,3,3)
    fx = K[..., 0, 0][..., None, None]
    fy = K[..., 1, 1][..., None, None]
    cx = K[..., 0, 2][..., None, None]
    cy = K[..., 1, 2][..., None, None]
    return fx, fy, cx, cy


def backproject_rgbd_to_world_points(
    rgb: torch.Tensor,        # (B,V,3,H,W) in [0,1] (or any range; we just carry it)
    depth: torch.Tensor,      # (B,V,1,H,W) depth in meters (or your unit), z in camera frame
    c2w: torch.Tensor,        # (B,V,4,4) camera-to-world
    K: torch.Tensor,          # (B,V,3,3) intrinsics in pixel units (or (B,3,3))
    *,
    depth_min: float = 1e-6,
    stride: int = 1,
    max_points: int | None = 500_000,
):
    """
    Returns:
      points_w: (B, N, 3)
      colors:   (B, N, 3)
      valid_mask: (B, N) boolean
    Notes:
      - Uses the standard pinhole backprojection with pixel coordinates (u,v).
      - If max_points is set, randomly subsamples PER BATCH ITEM after concatenating all views.
    """
    device = rgb.device
    B, V, _, H, W = rgb.shape

    if stride > 1:
        rgb = rgb[..., ::stride, ::stride]
        depth = depth[..., ::stride, ::stride]
        H = rgb.shape[-2]
        W = rgb.shape[-1]
    
    s = float(stride)
    fx, fy, cx, cy = _intrinsics_to_fxfycxcy(K.to(rgb.dtype))
    if stride > 1:
        fx = fx / s
        fy = fy / s
        cx = cx / s
        cy = cy / s

    # pixel grid (u to the right, v down), shape (1,1,H,W)
    u = torch.arange(W, device=device, dtype=rgb.dtype)[None, None, None, :]
    v = torch.arange(H, device=device, dtype=rgb.dtype)[None, None, :, None]
    u = u.expand(B, V, H, W)
    v = v.expand(B, V, H, W)

    z = depth[:, :, 0]  # (B,V,H,W)
    valid = z > depth_min

    # camera-frame points
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # (B,V,H,W,3)
    p_c = torch.stack([x, y, z], dim=-1)

    # transform to world: p_w = R * p_c + t
    R = c2w[:, :, :3, :3]  # (B,V,3,3)
    t = c2w[:, :, :3, 3]   # (B,V,3)

    p_c_flat = p_c.reshape(B, V, H * W, 3)
    p_w_flat = torch.einsum("bvij,bvnj->bvni", R, p_c_flat) + t[:, :, None, :]  # (B,V,HW,3)

    colors_flat = rgb.permute(0, 1, 3, 4, 2).reshape(B, V, H * W, 3)  # (B,V,HW,3)
    valid_flat = valid.reshape(B, V, H * W)  # (B,V,HW)

    # concatenate views -> (B, N, 3)
    p_w = p_w_flat.reshape(B, V * H * W, 3)
    c = colors_flat.reshape(B, V * H * W, 3)
    m = valid_flat.reshape(B, V * H * W)

    # keep only valid via mask later (preserve shape for easier batching)
    # optional subsample
    if max_points is not None:
        # sample indices per batch from valid points
        out_p = []
        out_c = []
        out_m = []
        for b in range(B):
            idx_valid = torch.nonzero(m[b], as_tuple=False).squeeze(-1)
            if idx_valid.numel() == 0:
                # no valid points
                out_p.append(p_w[b:b+1, :1] * 0.0)
                out_c.append(c[b:b+1, :1] * 0.0)
                out_m.append(torch.zeros((1, 1), device=device, dtype=torch.bool))
                continue

            if idx_valid.numel() > max_points:
                perm = torch.randperm(idx_valid.numel(), device=device)[:max_points]
                idx = idx_valid[perm]
            else:
                idx = idx_valid

            out_p.append(p_w[b:b+1, idx])
            out_c.append(c[b:b+1, idx])
            out_m.append(torch.ones((1, idx.numel()), device=device, dtype=torch.bool))

        points_w = torch.cat(out_p, dim=0)
        colors = torch.cat(out_c, dim=0)
        valid_mask = torch.cat(out_m, dim=0)
        return points_w, colors, valid_mask

    return p_w, c, m


def render_orthographic_pointcloud_projections(
    points_w: torch.Tensor,   # (B,N,3)
    colors: torch.Tensor,     # (B,N,3)
    valid_mask: torch.Tensor, # (B,N) bool
    resolution: int,
    margin: float = 0.1,
    background: float = 0.0,
    eps: float = 1e-6,
):
    """
    Produces 3 orthographic projections by axis-aligned "cameras".
    Returns: (B, 3, 3, resolution, resolution)
      - second dim indexes the 3 looks (look_axis = 0,1,2)
      - each is RGB
    """
    device = points_w.device
    B, N, _ = points_w.shape

    # AABB per batch
    # (ignore invalid by setting them far away)
    big = 1e9
    pw = points_w.clone()
    pw[~valid_mask] = big
    minima = pw.min(dim=1).values
    pw2 = points_w.clone()
    pw2[~valid_mask] = -big
    maxima = pw2.max(dim=1).values

    scene_minima, scene_maxima = compute_equal_aabb_with_margin(minima, maxima, margin=margin)

    projections = []
    for look_axis in range(3):
        right_axis = (look_axis + 1) % 3
        down_axis = (look_axis + 2) % 3

        # coordinates along plane axes
        x = points_w[..., right_axis]  # (B,N)
        y = points_w[..., down_axis]   # (B,N)
        z = points_w[..., look_axis]   # (B,N) used for z-buffer (near = smaller z)

        # normalize to pixel coordinates [0, res-1]
        x0 = scene_minima[:, right_axis][:, None]
        x1 = scene_maxima[:, right_axis][:, None]
        y0 = scene_minima[:, down_axis][:, None]
        y1 = scene_maxima[:, down_axis][:, None]

        # avoid div0
        x_den = (x1 - x0).clamp_min(eps)
        y_den = (y1 - y0).clamp_min(eps)

        u = (x - x0) / x_den * (resolution - 1)
        v = (y - y0) / y_den * (resolution - 1)

        # integer pixel coords
        ui = u.round().clamp(0, resolution - 1).to(torch.long)
        vi = v.round().clamp(0, resolution - 1).to(torch.long)

        # flatten pixel index
        pix = vi * resolution + ui  # (B,N)
        P = resolution * resolution

        # z-buffer: find min z for each pixel
        z_init = torch.full((B, P), float("inf"), device=device, dtype=points_w.dtype)

        # torch.scatter_reduce available in recent PyTorch (2.0+)
        zmin = z_init.scatter_reduce(
            dim=1,
            index=pix,
            src=torch.where(valid_mask, z, torch.full_like(z, float("inf"))),
            reduce="amin",
            include_self=True,
        )  # (B,P)

        # points that are at the min depth for their pixel (within tolerance)
        zmin_at_point = torch.gather(zmin, 1, pix)  # (B,N)
        keep = valid_mask & (z <= zmin_at_point + 1e-4)  # tolerance helps with quantization

        # accumulate color + counts
        img = torch.full((B, 3, P), background, device=device, dtype=colors.dtype)
        cnt = torch.zeros((B, 1, P), device=device, dtype=colors.dtype)

        # scatter-add colors
        for ch in range(3):
            src = torch.where(keep, colors[..., ch], torch.zeros_like(colors[..., ch]))
            img[:, ch].scatter_add_(1, pix, src)

        cnt[:, 0].scatter_add_(1, pix, keep.to(colors.dtype))

        img = img / cnt.clamp_min(1.0)  # average if multiple points hit same pixel

        img = img.reshape(B, 3, resolution, resolution)
        projections.append(img)

    return torch.stack(projections, dim=1)  # (B, 3, 3, res, res)


# ------------------------------------------------------------
# Main wrapper (RGBD -> point cloud -> 3 projections)
# ------------------------------------------------------------

def render_pointcloud_projections_from_rgbd(
    rgb: torch.Tensor,        # (B,V,3,H,W)
    depth: torch.Tensor,      # (B,V,1,H,W)
    c2w: torch.Tensor,        # (B,V,4,4)
    K: torch.Tensor,          # (B,V,3,3) intrinsics in pixel units (or (B,3,3))
    resolution: int,
    *,
    margin: float = 0.1,
    stride: int = 2,
    max_points: int | None = 500_000,
    background: float = 0.0,
    draw_label: bool = False,
    extra_label: str = "",
):
    """
    Returns: (B, 3, 3, resolution, resolution)
    """
    points_w, colors, valid = backproject_rgbd_to_world_points(
        rgb=rgb,
        depth=depth,
        c2w=c2w,
        K=K,
        stride=stride,
        max_points=max_points,
    )

    proj = render_orthographic_pointcloud_projections(
        points_w=points_w,
        colors=colors,
        valid_mask=valid,
        resolution=resolution,
        margin=margin,
        background=background,
    )

    # Optional labeling hook (kept compatible with your existing add_label)
    if draw_label:
        # expects add_label(image_chw, text) -> image_chw
        # and operates per projection
        labels = []
        for look_axis in range(3):
            right_axis = (look_axis + 1) % 3
            down_axis = (look_axis + 2) % 3
            label = f'{"XYZ"[right_axis]}{"XYZ"[down_axis]} Projection {extra_label}'.strip()
            labels.append(torch.stack([add_label(x, label) for x in proj[:, look_axis]]))
        proj = torch.stack(labels, dim=1)

    return proj