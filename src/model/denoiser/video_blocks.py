

from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.downsampling import (  # noqa
    Downsample1D,
    Downsample2D,
    FirDownsample2D,
    KDownsample2D,
    downsample_2d,
)
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.upsampling import (  # noqa
    FirUpsample2D,
    KUpsample2D,
    Upsample1D,
    Upsample2D,
    upfirdn2d_native,
    upsample_2d,
)

class RayAdaLN(nn.Module):
    """
    Spatial AdaLN from ray maps.
    rays: (B, Cr, T, H, W)
    returns shift, scale: (B, C, T, H, W)
    """
    def __init__(self, ray_channels: int, feat_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ray_channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden, 2 * feat_channels, kernel_size=1),
        )
        # AdaLN-Zero: start with shift=0, scale=0 so modulation is identity
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, rays_bcthw: torch.Tensor):
        shift_scale = self.net(rays_bcthw)  # (B, 2C, T, H, W)
        shift, scale = shift_scale.chunk(2, dim=1)
        return shift, scale

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TemporalResnetBlockAdaLN(nn.Module):
    r"""
    A Resnet block with adaptive layer-normalization for conditioning on camera poses.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        temb_channels: int | None = None,
        ray_channels: int = 6, 
        ray_hidden: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.nonlinearity = get_activation("silu")

        # Two AdaLN heads (norm1 and norm2)
        # Ray-conditioned AdaLN modulators (spatial!)
        self.adaln1 = RayAdaLN(ray_channels=ray_channels, feat_channels=in_channels, hidden=ray_hidden)
        self.adaln2 = RayAdaLN(ray_channels=ray_channels, feat_channels=out_channels, hidden=ray_hidden)

        # Make residual branch no-op at init.
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, input_tensor: torch.Tensor, plucker_rays: dict[tuple, torch.Tensor], temb: torch.Tensor | None = None) -> torch.Tensor:
        # input_tensor: (B, C, T, H, W)
        # plucker_rays: (B, T, Cr, H, W)
        b, c, t, h, w = input_tensor.shape
        if (temb is not None) and len(temb.shape) == 2:
            assert temb.shape[0] == b * t
            temb = rearrange(temb, "(b t) d -> b t d", b=b, t=t)

        hidden_states = input_tensor

        rays = plucker_rays[(h,w)]
        # rays to (B, Cr, T, H, W) to match Conv3d convention
        rays = rays.permute(0, 2, 1, 3, 4).contiguous()

        hidden_states = self.norm1(hidden_states)
        shift1, scale1 = self.adaln1(rays)
        hidden_states = modulate(hidden_states, shift1, scale1)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            assert temb is not None
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        shift2, scale2 = self.adaln2(rays)
        hidden_states = modulate(hidden_states, shift2, scale2)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return hidden_states
