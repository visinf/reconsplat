import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..types import AnyExample, AnyViews
from ...misc.camera_utils import normalize_K

def crop_and_rescale(
    images: torch.Tensor,          # [..., C, H, W]
    intrinsics: torch.Tensor,      # [..., 3, 3]
    out_shape: tuple[int, int],    # (h_out, w_out)
    depths: torch.Tensor | None = None,  # [..., H, W]
    masks: torch.Tensor | None = None, # [..., H, W]
):
    """
    Center-crop to the target aspect ratio and rescale to (h_out, w_out),
    updating intrinsics accordingly. If depth maps are provided, crop them as well.
    """
    assert images.ndim >= 4 and images.shape[-3] > 0, "images must be [..., C, H, W]"
    assert intrinsics.shape[-2:] == (3, 3), "intrinsics must be [..., 3, 3]"
    if depths is not None:
        assert images.shape[-2:] == depths.shape[-2:], "images and depths shapes should be the same"
    h_out, w_out = int(out_shape[0]), int(out_shape[1])
    assert h_out > 0 and w_out > 0, "out_shape must be positive"

    *batch, C, H, W = images.shape
    device = images.device
    dtype  = images.dtype

    target_ar = w_out / h_out
    in_ar = W / H

    #  Determine crop size (Hc, Wc) that matches the target aspect ratio.
    if in_ar > target_ar:
        # Input is "wider", so crop width.
        Hc = H
        Wc = int(round(H * target_ar))
        Wc = max(1, min(Wc, W))
        top = 0
        left = (W - Wc) // 2
    else:
        # Input is "taller", so crop height.
        Wc = W
        Hc = int(round(W / target_ar))
        Hc = max(1, min(Hc, H))
        left = 0
        top = (H - Hc) // 2

    # Center-crop images.
    images_c = images[..., :, top:top+Hc, left:left+Wc]
    # If available, center-crop depth.
    if depths is not None:
        depths_c = depths[..., top:top+Hc, left:left+Wc]
        if masks is not None:
            masks_c = masks[..., top:top+Hc, left:left+Wc]

    # Adjusts intriniscs to account for the crop.
    intrinsics_adj = intrinsics.clone()
    intrinsics_adj[..., 0, 2] -= left
    intrinsics_adj[..., 1, 2] -= top

    # Scale factors in x,y w.r.t. the CROP size.
    sx = w_out / Wc
    sy = h_out / Hc

    assert np.abs(sx - sy) < 1e-3, "We expect the scaling factors for intrinsics to be the same sx==sy at this point."

    images_c = images_c.reshape(-1, C, Hc, Wc)
    images_r = F.interpolate(images_c, size=(h_out, w_out), mode="bilinear", align_corners=False)
    images_r = images_r.reshape(*batch, C, h_out, w_out)

    # Scale intrinsics (fx, fy, cx, cy)
    intrinsics_adj[..., 0, 0] *= sx
    intrinsics_adj[..., 1, 1] *= sy
    intrinsics_adj[..., 0, 2] *= sx
    intrinsics_adj[..., 1, 2] *= sy

    if depths is not None:
        depths_c = depths_c.reshape(-1, 1, Hc, Wc)
        depths_r = F.interpolate(depths_c, size=(h_out, w_out), mode="bilinear", align_corners=False)
        depths_out = depths_r.squeeze(1)
        if masks is not None:
            masks_c = masks_c.reshape(-1, 1, Hc, Wc)
            masks_r = F.interpolate(masks_c, size=(h_out, w_out), mode="bilinear", align_corners=False)
            masks_out = masks_r.squeeze(1)
        else:
            masks_out = None

    return (images_r, intrinsics_adj, depths_out, masks_out) if depths is not None else (images_r, intrinsics_adj)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int], normalize_intrinsics: bool = True) -> AnyViews:
    h_out, w_out = shape

    if 'depth' in views.keys():
        images, intrinsics, depths, masks = crop_and_rescale(
            views["image"],
            views["intrinsics"],
            shape,
            depths=views["depth"],
            masks=views["mask"] if "mask" in views else None
        )
    else:
        images, intrinsics = crop_and_rescale(views["image"], views["intrinsics"], shape)
        depths = None
        masks = None

    intrinsics = normalize_K(intrinsics, w_out, h_out) if normalize_intrinsics else intrinsics

    crop_example = {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }
    # Only carry depth when the views actually had it; writing "depth": None back would defeat the
    # point, since the default collate cannot handle a None.
    if depths is not None:
        crop_example["depth"] = depths

    if "mask" in views: # not available for test or val.
        crop_example["mask"] = masks
    
    return crop_example

def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
