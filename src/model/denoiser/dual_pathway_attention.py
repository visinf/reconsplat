from dataclasses import dataclass
from typing import Literal

import copy
import torch
from torch import nn
from einops import rearrange

from ..attention.prope import PRoPECrossAttention
from ...misc.nn_module_tools import _copy_linear, _copy_tensor
from .mvdream.attention import BasicTransformerBlock, Normalize, exists, zero_module
from .video_blocks import TemporalResnetBlockAdaLN


def _first_linear(container: nn.Module) -> nn.Linear:
    # Handles ModuleList([Linear, Dropout]) and Sequential(Linear, Dropout)
    if isinstance(container, (nn.ModuleList, nn.Sequential)):
        if len(container) == 0:
            raise ValueError("Empty container where Linear was expected.")
        if not isinstance(container[0], nn.Linear):
            raise TypeError(f"Expected first element to be Linear, got {type(container[0])}")
        return container[0]
    # Sometimes it's directly a Linear
    if isinstance(container, nn.Linear):
        return container
    raise TypeError(f"Unsupported container type: {type(container)}")


@dataclass
class SpatialTransformer3DCfg:
    name: Literal["spatial_transformer_3d"]
    num_heads: int
    num_layers: int = 1
    d_dot: int | None = None                # default d_in // num_heads
    d_mlp: int | None = None                # if None: default d_in * d_mlp_multiplier
    d_mlp_multiplier: int | None = None
    downscale: int = 1


# NOTE: "dual" refers to the two attention branches: 2D (single-view, attn1) and 3D (multi-view, attn3d).
# Both are always active. The 3D conv branch (temp3d), despite its name conditioned on cameras rather
# than time, is separate from these two and only active when fine-tuning on sequences with the rest
# of the network frozen; otherwise it is a no-op.
class DualTransformerBlock3D(BasicTransformerBlock):
    def __init__(self, dim, image_width, image_height, latent_width, latent_height,
                 n_heads, d_head, dropout=0, context_dim=None,
                 gated_ff=True, checkpoint=True, disable_self_attn=False, enable_temporal_convs=False):
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, checkpoint, disable_self_attn)
        print("[NOTE - DualTransformerBlock3D] Using softmax-prope attention.")
        self.image_width = image_width
        self.image_height = image_height
        self.latent_width = latent_width
        self.latent_height = latent_height

        self.attn3d = PRoPECrossAttention(
            query_dim=dim,
            image_width=image_width,
            image_height=image_height,
            latent_width=latent_width,
            latent_height=latent_height,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None
        )
        self.norm3d = copy.deepcopy(self.norm1)
        self.zero_proj_3d = zero_module(nn.Linear(dim, dim, bias=False))

        if enable_temporal_convs:
            # NOTE: temporal residual branch - only active if fine-tuning on sequences.
            self.temp3d = TemporalResnetBlockAdaLN(in_channels=dim)
        else:
            self.temp3d = None

    def forward(self, x, extr, intr, plucker_cache=None, context=None, num_frames=1):
        return self._forward(x, extr, intr, plucker_cache, context, num_frames)

    def _forward(self, x, extr, intr, plucker_cache=None, context=None, num_frames=1):
        residual = x

        # 2D branch
        delta_2d = self.attn1(
            self.norm1(residual),
            context=context if self.disable_self_attn else None
        )

        # 3D branch
        x_3d_in = rearrange(residual, "(b f) l c -> b (f l) c", f=num_frames).contiguous()
        x_3d_in = self.norm3d(x_3d_in)
        delta_3d = self.attn3d(
            x_3d_in,
            viewmats=extr,
            Ks=intr,
            context=context if self.disable_self_attn else None
        )
        delta_3d = self.zero_proj_3d(delta_3d)
        delta_3d = rearrange(delta_3d, "b (f l) c -> (b f) l c", f=num_frames).contiguous()

        # combine both residual branches
        x = residual + delta_2d + delta_3d

        # temporal branch - if active
        if self.temp3d is not None:
            assert plucker_cache is not None, "plucker_cache, pre-computed from extrinsics and intrinsics, must be provided for temporal convs."
            # residual here is not pre-normalized with LayerNorm, unlike the attn1/attn3d branches above:
            # temp3d operates on conv features, for which GroupNorm is the more suitable normalization
            # (grouped per-channel, spatial) than LayerNorm (per-token, used for attention branches).
            # It applies its own GroupNorm + AdaLN internally before each conv, so no pre-norm is needed.
            delta_temp = rearrange(
                residual,
                "(b f) (h w) c -> b c f h w",
                f=num_frames,
                h=self.latent_height,
                w=self.latent_width,
            )
            delta_temp = self.temp3d(delta_temp, plucker_rays=plucker_cache)
            delta_temp = rearrange(
                delta_temp,
                "b c f h w -> (b f) (h w) c",
            )
            x = x + delta_temp

        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer3D(nn.Module):
    ''' 3D self-attention '''
    def __init__(self, cfg: SpatialTransformer3DCfg, d_in: int,
                 image_width=None, image_height=None, latent_width=None, latent_height=None,
                 dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,
                 orig_unet_attn_block=None,
                 enable_temporal_convs=False
                 ):

        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]

        if not exists(context_dim):
            context_dim = [context_dim] * cfg.num_layers

        in_channels = d_in
        n_heads = cfg.num_heads
        d_head = cfg.d_dot or in_channels // n_heads
        self.in_channels = in_channels
        inner_dim = in_channels
        depth = cfg.num_layers
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [DualTransformerBlock3D(inner_dim,
                                image_width, image_height, latent_width, latent_height,
                                n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                disable_self_attn=disable_self_attn, enable_temporal_convs=enable_temporal_convs, checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

        if orig_unet_attn_block is not None:
            # Init from attn pre-trained weights.
            self.init_from_block(orig_unet_attn_block)

    @torch.no_grad()
    def init_from_block(self, block: nn.Module):
        # Norm
        _copy_tensor(block.norm.weight, self.norm.weight, "norm.weight")
        _copy_tensor(block.norm.bias,   self.norm.bias,   "norm.bias")
        # linear layers
        _copy_linear(block.proj_in, self.proj_in, "proj_in")
        _copy_linear(block.proj_out, self.proj_out, "proj_out")
        # sub blocks (3x at each resolution)
        n = min(len(block.transformer_blocks), len(self.transformer_blocks))
        if len(block.transformer_blocks) != len(self.transformer_blocks):
            print(f"[warn] different number of blocks: src={len(block.transformer_blocks)} dst={len(self.transformer_blocks)}; copying first {n}")

        for i in range(n):
            # source block, destination block, and path prefix
            sb = block.transformer_blocks[i]
            db = self.transformer_blocks[i]
            p = f"transformer_blocks[{i}]"

            # norms 1/2/3
            print(f"Copy: {p}.norm1/2/3")
            for k in (1, 2, 3):
                sn = getattr(sb, f"norm{k}")
                dn = getattr(db, f"norm{k}")
                _copy_tensor(sn.weight, dn.weight, f"{p}.norm{k}.weight")
                _copy_tensor(sn.bias,   dn.bias,   f"{p}.norm{k}.bias")

            # attn1 qkv projections
            print(f"Copy: {p}.attn1 qkv + out")
            for name in ("to_q", "to_k", "to_v"):
                _copy_linear(getattr(sb.attn1, name), getattr(db.attn1, name), f"{p}.attn1.{name}")
            # attn1 out proj
            _copy_linear(_first_linear(sb.attn1.to_out), _first_linear(db.attn1.to_out), f"{p}.attn1.to_out.0")

            # if the transformer block as a dual attn branch (single- and multi-view), initialize attn3d with attn1 weights
            if hasattr(db, 'attn3d'):
                print(f"Copy: {p}.attn3d qkv + out")
                for name in ("to_q", "to_k", "to_v"):
                    _copy_linear(getattr(sb.attn1, name), getattr(db.attn3d, name), f"{p}.attn3d.{name}")
                # attn3d out proj
                _copy_linear(_first_linear(sb.attn1.to_out), _first_linear(db.attn3d.to_out), f"{p}.attn3d.to_out.0")

            # attn2 qkv projections
            print(f"Copy: {p}.attn2 qkv + out")
            for name in ("to_q", "to_k", "to_v"):
                _copy_linear(getattr(sb.attn2, name), getattr(db.attn2, name), f"{p}.attn2.{name}")
            _copy_linear(_first_linear(sb.attn2.to_out), _first_linear(db.attn2.to_out), f"{p}.attn2.to_out.0")

            # FFN: GEGLU proj + output linear
            print(f"Copy: {p}.ff (GEGLU proj + out)")
            # src: sb.ff.net is ModuleList([GEGLU, Dropout, Linear])
            # dst: db.ff.net is Sequential([GEGLU, Dropout, Linear])
            s_geglu = sb.ff.net[0]
            d_geglu = db.ff.net[0]
            _copy_linear(s_geglu.proj, d_geglu.proj, f"{p}.ff.net.0.proj")

            s_ff_out = sb.ff.net[2]
            d_ff_out = db.ff.net[2]
            _copy_linear(s_ff_out, d_ff_out, f"{p}.ff.net.2")

    def forward(self, x, extr=None, intr=None, plucker_cache=None, context=None):
        # intr might be none and fall back to GTA-style attention
        assert extr is not None

        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, v, c, h, w = x.shape
        num_frames = v
        x = rearrange(x, 'b v c h w -> (b v) c h w').contiguous()
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, extr=extr, intr=intr, plucker_cache=plucker_cache, context=context[i], num_frames=num_frames)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        res = x + x_in
        return rearrange(res, "(b v) c h w -> b v c h w", v=num_frames)
