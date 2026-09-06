from ..types import BatchedExample, BatchedViews


def apply_patch_shim_to_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    _, _, _, h, w = views["image"].shape

    # Image size must be even so that naive center-cropping does not cause misalignment.
    assert h % 2 == 0 and w % 2 == 0

    h_new = (h // patch_size) * patch_size
    row = (h - h_new) // 2
    w_new = (w // patch_size) * patch_size
    col = (w - w_new) // 2

    # Center-crop the image.
    image = views["image"][:, :, :, row : row + h_new, col : col + w_new]
    if "depth" in views:
        depth = views["depth"][:, :, row : row + h_new, col : col + w_new]
        if "mask" in views:
            mask = views["mask"][:, :, row : row + h_new, col : col + w_new]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    intrinsics[:, :, 0, 0] *= w / w_new  # fx
    intrinsics[:, :, 1, 1] *= h / h_new  # fy

    out_example =  {
        **views,
        "image": image,
        "intrinsics": intrinsics,
    }
    # Conditional for the same reason as in crop_shim -- and because `depth` is only bound inside the
    # "depth" in views branch above, so referencing it unconditionally raised NameError without labels.
    if "depth" in views:
        out_example["depth"] = depth
    if "mask" in views: # not available for val or test.
        out_example["mask"] = mask
    return out_example


def apply_patch_shim(batch: BatchedExample, patch_size: int) -> BatchedExample:
    """Crop images in the batch so that their dimensions are cleanly divisible by the
    specified patch size.
    """
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }
