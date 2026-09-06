import io
from pathlib import Path
from typing import Union
import skvideo.io

import imageio
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]
ArrayLike = Union[np.ndarray, "torch.Tensor"]

def _to_uint8_rgb(frame: ArrayLike) -> np.ndarray:
    """Normalize/convert an input frame to HxWx3 uint8 RGB."""
    if isinstance(frame, torch.Tensor):
        # if on GPU, bring to CPU
        if frame.is_cuda:
            frame = frame.detach().cpu()
        frame = frame.detach()
        if torch.is_floating_point(frame):
            # numpy has no bfloat16/float16 equivalent; normalize to float32 before converting.
            frame = frame.float()
        npf = frame.numpy()
    else:
        npf = np.asarray(frame)

    # Handle channel-first vs channel-last
    # Accept shapes: (H, W), (H, W, 1), (H, W, 3), (C, H, W)
    if npf.ndim == 2:
        # grayscale -> RGB
        npf = npf[..., None].repeat(3, axis=-1)  # (H, W, 3)
    elif npf.ndim == 3:
        if npf.shape[0] in (1, 3) and npf.shape[-1] not in (1, 3):
            # likely CHW -> HWC
            npf = np.transpose(npf, (1, 2, 0))
        if npf.shape[-1] == 1:
            npf = np.repeat(npf, 3, axis=-1)
        elif npf.shape[-1] != 3:
            raise ValueError(f"Unsupported frame shape {npf.shape}: expected last dim 1 or 3.")
    else:
        raise ValueError(f"Unsupported frame ndim {npf.ndim}: expected 2 or 3.")

    # Convert to uint8 [0,255]
    if npf.dtype == np.uint8:
        out = npf
    else:
        # Try to infer range; if float, assume [0,1] or [-1,1]
        if np.issubdtype(npf.dtype, np.floating):
            fmin, fmax = npf.min(), npf.max()
            # Heuristic: if values in [-1,1], map to [0,1]; otherwise clip [0,1]
            if fmin >= -1.0 and fmax <= 1.0:
                npf = (npf + 1.0) * 0.5
            npf = np.clip(npf, 0.0, 1.0)
            out = (npf * 255.0 + 0.5).astype(np.uint8)
        else:
            # integer or other -> clip to [0,255]
            out = np.clip(npf, 0, 255).astype(np.uint8)

    # Ensure C-contiguous
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)

    return out

def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    images: list[FloatImage],
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    # Image.fromarray(prep_image(image)).save(path)
    frames = []
    for image in images:
        frames.append(prep_image(image))

    writer = skvideo.io.FFmpegWriter(path, 
                                     outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', 
                                                 '-vf': f'setpts=1.*PTS'})
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()

def save_video_mp4(
        images: list[FloatImage],
        path: Union[Path, str],
        fps=10
)  -> None:

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    with imageio.imopen(str(path), io_mode="w") as w:
        for image in images:
            rgb = _to_uint8_rgb(image)
            w.write(rgb)

def render_depth_map(depth, _min=None, _max=None, cmap="magma"):
    """
    Render a depth map tensor (H, W) as a colored RGB image.    
    Args:
        depth (torch.Tensor): Tensor of shape (H, W)
        cmap (str): Colormap name (default: 'magma')
    
    Returns:
        torch.Tensor: Colored image of shape (3, H, W) in [0, 1]
    """
    depth = depth.detach().float().cpu()
    # Normalize to [0,1]
    if _min is None and _max is None:
        # per-depthmap normalization
        depth_min, depth_max = depth.min(), depth.max()
    else:
        depth_min, depth_max = _min, _max

    if depth_max > depth_min:  # avoid div by zero
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)
    
    # Apply matplotlib colormap
    colormap = plt.get_cmap(cmap)
    depth_colored = colormap(depth_norm)[:, :, :3]  # ignore alpha channel
    # Back to torch tensor with shape (3, H, W)
    depth_colored_tensor = torch.from_numpy(depth_colored).permute(2, 0, 1).float()
    return depth_colored_tensor

def render_depth_map_percentile(depth, cmap="Spectral"):
    """
    Render a depth map tensor (H, W) as a colored RGB image, with percentile filtering of outliers.    
    Args:
        depth (torch.Tensor): Tensor of shape (H, W)
        cmap (str): Colormap name (default: 'Spectral')
    
    Returns:
        torch.Tensor: Colored image of shape (3, H, W) in [0, 1]
    """
    depth = depth.detach().float().cpu().numpy()
    vmax = np.percentile(depth, 95)
    vmin = np.percentile(depth, 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3]).astype(
        np.float32
    )  # [H, W, 3]
    colormapped_im = torch.from_numpy(colormapped_im).permute(2, 0, 1).float()
    return colormapped_im

def render_error_map(error_map: torch.Tensor, cmap: str = "magma", vmin: float = None, vmax: float = None):
    """
    Render an error map (e.g., pixel or depth reprojection error) as a colored RGB image.
    
    Args:
        error_map (torch.Tensor): Tensor of shape (H, W) or (1, H, W)
        cmap (str): Matplotlib colormap (default: 'magma')
        vmin (float, optional): Minimum value for normalization.
        vmax (float, optional): Maximum value for normalization.
    
    Returns:
        torch.Tensor: Colored image tensor of shape (3, H, W) in [0, 1]
    """
    if error_map.ndim == 3 and error_map.shape[0] == 1:
        error_map = error_map.squeeze(0)
    error_map = error_map.detach().cpu().float()

    # Compute normalization bounds
    if vmin is None:
        vmin = float(error_map.min())
    if vmax is None:
        vmax = float(error_map.max())

    # Prevent degenerate normalization
    if vmax > vmin:
        normed = (error_map - vmin) / (vmax - vmin)
    else:
        normed = torch.zeros_like(error_map)

    # Apply colormap
    cmap_func = plt.get_cmap(cmap)
    colored = cmap_func(normed.numpy())[:, :, :3]  # drop alpha
    colored_tensor = torch.from_numpy(colored).permute(2, 0, 1).float()

    return colored_tensor