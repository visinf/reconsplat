from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor


def convert_images(
    images: list[UInt8[Tensor, "..."]],
) -> Float[Tensor, "batch 3 height width"]:
    torch_images = []
    transform = tf.ToTensor()
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(transform(image))
    return torch.stack(torch_images)

def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")

def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: None | Float[Tensor, "*#batch h w"] = None,
) -> (
    tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
    | tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
        Float[Tensor, "*#batch h_out w_out"],  # updated depths
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.

    # First denomalize.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in
    intrinsics[..., 1, 1] *= h_in
    intrinsics[..., 0, 2] *= w_in
    intrinsics[..., 1, 2] *= h_in

    # crop shift: principal point moves by (col, row)
    intrinsics[..., 0, 2] -= col
    intrinsics[..., 1, 2] -= row

    # renomalize to the new image size
    intrinsics[..., 0, 0] /= w_out  # fx
    intrinsics[..., 1, 1] /= h_out  # fy
    intrinsics[..., 0, 2] /= w_out  # cx
    intrinsics[..., 1, 2] /= h_out  # cy

    if depths is not None:
        depths = depths[..., :, row : row + h_out, col : col + w_out]
        return images, intrinsics, depths

    return images, intrinsics

def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: None | Float[Tensor, "*#batch h_d w_d"] = None,
) -> (
    tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
    | tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
        Float[Tensor, "*#batch h_out w_out"],  # updated depths
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the normalized intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    # reshape and crop depth as well when available
    if depths is not None:
        depths = torch.nn.functional.interpolate(
            depths.unsqueeze(1),
            size=(h_scaled, w_scaled),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

    return center_crop(images, intrinsics, shape, depths=depths)
