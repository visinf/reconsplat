import torch
import os 
from torch import Tensor, nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float, Int64
from typing import Literal, Optional, Tuple
from einops import rearrange, repeat
from diffusers import UNet2DConditionModel
from itertools import chain

from .denoiser import Denoiser
from peft import LoraConfig, get_peft_model
from ...misc.camera_utils import denormalize_K
from .attention import MultiViewAttentionCfg
from .dual_pathway_attention import SpatialTransformer3D
from .video_blocks import TemporalResnetBlockAdaLN
import torch.nn.functional as F

DISABLE_TORCH_COMPILE = int(os.getenv('DISABLE_TORCH_COMPILE', True))
if DISABLE_TORCH_COMPILE == 0:
    DISABLE_TORCH_COMPILE = False
else:
    DISABLE_TORCH_COMPILE = True
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False

@dataclass
class UNet2DModelCfg:
    name: Literal["unet"]
    down_block_types: list | Tuple
    mid_block_type: str
    up_block_types: list | Tuple
    only_cross_attention: bool
    block_out_channels: list | Tuple

@dataclass
class MultiViewUNetCfg:
    name: Literal["mv_unet"]
    autoencoder: UNet2DModelCfg
    multi_view_attention: MultiViewAttentionCfg
    use_ray_encoding: bool=True
    encoder_conditioning: bool=True  
    mid_conditioning: bool=True  
    decoder_conditioning: bool=True  
    temporal_conditioning: bool=False
    gradient_checkpointing: bool = False  # Trade compute for activation memory.
    pretrained_from: str | None = None
    pretrained_revision: str | None = None
    use_prope_encoding: bool = True
    video_finetuning: bool = False
    disable_rast_cond: bool = False # Ablation 1.
    low_res_cross_view_attn: bool = True
    min_scale_factor_cross_view_attn: int = 8

def zero_like_(t: torch.Tensor):
    with torch.no_grad():
        t.zero_()

def inflate_conv_(new_w: torch.Tensor, old_w: torch.Tensor):
    with torch.no_grad():
        zero_like_(new_w)
        cout = min(new_w.shape[0], old_w.shape[0])
        cin  = min(new_w.shape[1], old_w.shape[1])
        kh   = min(new_w.shape[2], old_w.shape[2])
        kw   = min(new_w.shape[3], old_w.shape[3])
        new_w[:cout, :cin, :kh, :kw].copy_(old_w[:cout, :cin, :kh, :kw])

def inflate_bias_(new_b: torch.Tensor, old_b: torch.Tensor):
    if new_b is None or old_b is None:
        return
    with torch.no_grad():
        zero_like_(new_b)
        n = min(new_b.shape[0], old_b.shape[0])
        new_b[:n].copy_(old_b[:n])

class ZeroConv2d(nn.Conv2d):
    """Conv2d with weights/bias initialized to zero (ControlNet-style)."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class MultiViewUNet(Denoiser[MultiViewUNetCfg]):
    def __init__(
        self, 
        cfg: MultiViewUNetCfg,
        in_channels: int,
        out_channels: int,
        image_height: int, 
        image_width: int,
        upsample_latents: bool = False
    ) -> None:
        super().__init__(cfg)
        self.use_ray_encoding = cfg.use_ray_encoding
        self.pretrained_from = cfg.pretrained_from
        self.pretrained_revision = cfg.pretrained_revision
        self.upsample_latents = upsample_latents
        if self.pretrained_from is None:
            self.unet = UNet2DConditionModel(
                in_channels=in_channels, 
                out_channels=out_channels,
                down_block_types=cfg.autoencoder.down_block_types,
                mid_block_type=cfg.autoencoder.mid_block_type,
                up_block_types=cfg.autoencoder.up_block_types,
                only_cross_attention=cfg.autoencoder.only_cross_attention,
                block_out_channels=cfg.autoencoder.block_out_channels,
                cross_attention_dim=cfg.autoencoder.block_out_channels
            )
        else:
            print("Loading from Pretrained: ", self.pretrained_from,
                  f"@ {self.pretrained_revision}" if self.pretrained_revision else "@ main (unpinned)")
            self.unet = UNet2DConditionModel.from_pretrained(
                self.pretrained_from, subfolder="unet", revision=self.pretrained_revision)

            dev   = next(self.unet.parameters()).device
            dtype = next(self.unet.parameters()).dtype

            old_conv_in_w = self.unet.conv_in.weight.detach().cpu()
            old_conv_in_b = self.unet.conv_in.bias.detach().cpu() if self.unet.conv_in.bias is not None else None
            old_conv_out_w = self.unet.conv_out.weight.detach().cpu()
            old_conv_out_b = self.unet.conv_out.bias.detach().cpu() if self.unet.conv_out.bias is not None else None

            if (in_channels == old_conv_in_w.shape[1]) and (out_channels == old_conv_out_w.shape[0]):
                # No need to replace/expand conv_in and conv_out layers - ablation without depth modeling.
                pass
            else:
                new_conv_in  = nn.Conv2d(in_channels,  cfg.autoencoder.block_out_channels[0], kernel_size=3, padding=1, stride=1, bias=True).to(dev, dtype)
                new_conv_out = nn.Conv2d(cfg.autoencoder.block_out_channels[0], out_channels, kernel_size=3, padding=1, stride=1, bias=True).to(dev, dtype)

                depth_conv_out_w = torch.zeros_like(old_conv_out_w, dtype=dtype)
                depth_conv_out_b = torch.zeros_like(old_conv_out_b, dtype=dtype)

                with torch.no_grad():
                    new_conv_in.weight.copy_(
                        torch.concat([old_conv_in_w, old_conv_in_w], dim=1) / 2 
                    )
                    new_conv_in.bias.copy_(old_conv_in_b)
                    new_conv_out.weight.copy_(
                        torch.concat([old_conv_out_w, depth_conv_out_w], dim=0)
                    )
                    new_conv_out.bias.copy_(
                        torch.concat([old_conv_out_b, depth_conv_out_b], dim=0)
                    )

                self.unet.conv_in  = new_conv_in.to(dev)
                self.unet.conv_out = new_conv_out.to(dev)
                print(f"[inflate] conv_in:  {tuple(new_conv_in.weight.shape)}  <-  {tuple(old_conv_in_w.shape)}")
                print(f"[inflate] conv_out: {tuple(new_conv_out.weight.shape)} <-  {tuple(old_conv_out_w.shape)}")

            if XFORMERS_IS_AVAILABLE:
                self.unet.enable_xformers_memory_efficient_attention()

        if cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        downscale_factor = 8 if not self.upsample_latents else 4
        latent_width, latent_height = (image_width // downscale_factor), (image_height // downscale_factor)
        latent_dims = [
            (latent_height, latent_width),
            (latent_height//2, latent_width//2),
            (latent_height//4, latent_width//4),
            (latent_height//8, latent_width//8)
        ]

        if self.cfg.temporal_conditioning:
            self.down_temporal_conv_blocks = []

        # Build U-Net blocks. 
        # Only insert multi-view (cross-view) attention at low-res feature levels:
        # its cost scales quadratically with (num_views * H/f * W/f), with f being the downsampling factor.
        # As f changes though the U-Net, we only insert cross-view attention when f > self.cfg.min_scale_factor_cross_view_attn, which is a hyperparameter.

        if self.cfg.encoder_conditioning:
            for l, down_block in enumerate(self.unet.down_blocks):

                if hasattr(down_block, 'attentions'):
                    for i, transformer_layer in enumerate(down_block.attentions):
                        latent_h, latent_w = latent_dims[l]
                        if (
                            self.cfg.low_res_cross_view_attn 
                            and (image_width // latent_w > self.cfg.min_scale_factor_cross_view_attn) 
                            and (image_height // latent_h > self.cfg.min_scale_factor_cross_view_attn)
                        ): 
                            spatial_transformer_3d_layer = SpatialTransformer3D(
                                cfg.multi_view_attention,
                                d_in=down_block.resnets[i].out_channels,
                                image_width=image_width, image_height=image_height, 
                                latent_width=latent_w, latent_height=latent_h,
                                context_dim=transformer_layer.transformer_blocks[0].attn2.to_k.in_features,
                                use_linear=True,
                                orig_unet_attn_block=transformer_layer,
                                enable_temporal_convs=self.cfg.temporal_conditioning
                            )
                            down_block.attentions[i] = spatial_transformer_3d_layer

        if self.cfg.mid_conditioning:
            for i, transformer_layer in enumerate(self.unet.mid_block.attentions):
                latent_h, latent_w = latent_dims[-1]
                spatial_transformer_3d_layer = SpatialTransformer3D(
                    cfg.multi_view_attention,
                    d_in=self.unet.mid_block.resnets[i].out_channels,
                    image_width=image_width, image_height=image_height, 
                    latent_width=latent_w, latent_height=latent_h,
                    context_dim=transformer_layer.transformer_blocks[0].attn2.to_k.in_features,
                    use_linear=True,
                    orig_unet_attn_block=transformer_layer,
                    enable_temporal_convs=self.cfg.temporal_conditioning
                )
                self.unet.mid_block.attentions[i] = spatial_transformer_3d_layer

        latent_dims.reverse()

        if self.cfg.decoder_conditioning:
            for l, up_block in enumerate(self.unet.up_blocks):

                if hasattr(up_block, 'attentions'):
                    for i, transformer_layer in enumerate(up_block.attentions):
                        latent_h, latent_w = latent_dims[l]
                        if (
                            self.cfg.low_res_cross_view_attn 
                            and (image_width // latent_w > self.cfg.min_scale_factor_cross_view_attn) 
                            and (image_height // latent_h > self.cfg.min_scale_factor_cross_view_attn)
                        ): 
                            spatial_transformer_3d_layer = SpatialTransformer3D(
                                cfg.multi_view_attention,
                                d_in=up_block.resnets[i].out_channels,
                                image_width=image_width, image_height=image_height, 
                                latent_width=latent_w, latent_height=latent_h,
                                context_dim=transformer_layer.transformer_blocks[0].attn2.to_k.in_features,
                                use_linear=True,
                                orig_unet_attn_block=transformer_layer,
                                enable_temporal_convs=self.cfg.temporal_conditioning
                            )
                            up_block.attentions[i] = spatial_transformer_3d_layer

        # Build a multi-scale feature pyramid to condition the UNet with rasterized features.
        # TODO: avoid hard-coding the channel dimensions here.
        if not self.cfg.disable_rast_cond:
            down_channels_per_level = [
                [320, 320, 320],        # 32x32
                [320, 640, 640],        # 16x16
                [640, 1280, 1280],      # 8x8
                [1280, 1280, 1280]      # 4x4
            ]
            residuals_per_level = 3
            mid_channels = 1280 
            num_down_levels = 4 # 4 down blocks + mid (bottleneck)

            self.residuals_per_level = residuals_per_level
            self.rasterizer_proj_conv_in = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=320,
                kernel_size=3,
                padding=1)
            
            flat_out = [c for lvl in down_channels_per_level for c in lvl]

            blocks = []
            prev_ch = 320
            for out_ch in flat_out:
                blocks.append(nn.Sequential(
                    nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(32, out_ch),
                    nn.SiLU()
                ))
                prev_ch = out_ch
            self.rasterizer_proj = nn.ModuleList(blocks)

            self.rasterizer_proj.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels,
                            mid_channels,
                            kernel_size=3,
                            padding=1),
                    nn.GroupNorm(32, mid_channels),
                    nn.SiLU()
                ) 
            )

            self.zero_conv_injects = nn.ModuleList([
                ZeroConv2d(
                    out_ch,
                    out_ch,
                    kernel_size=1
                ) for out_ch in flat_out
            ])
            self.zero_conv_injects.append(
                ZeroConv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=1
                )
            )

        if self.cfg.video_finetuning:
            # Disable gradients for all the weights, except temporal blocks.
            for param in self.parameters():
                param.requires_grad = False

            for module in self.unet.modules():
                if isinstance(module, TemporalResnetBlockAdaLN):
                    for param in module.parameters():
                        param.requires_grad = True
            
    def forward(
        self,
        latents:  Float[Tensor, "batch view _ height width"],
        timestep: Int64[Tensor, "batch ..."] | Float[Tensor, "batch ..."],
        cond_state: Optional[Tensor]=None,
        rfeatures: Optional[Float[Tensor, "batch view _ height width"]] = None,
        extrinsics: Optional[Float[Tensor, "batch view i i"]] = None,
        intrinsics: Optional[Float[Tensor, "batch view j j"]] = None,
        plucker_rays: dict[tuple[int, int], Float[Tensor, "batch view 6 ..."]] | None = None,
        unconditional: bool = False
    ) -> Float[Tensor, "batch view _ height width"]:

        b, num_views, c, h, w = latents.shape
        hidden_states = latents
    
        # Process timesteps
        if len(timestep.shape) < 2:
            timestep = repeat(timestep, "b ... -> (b v) ...", v=num_views)
        else:
            timestep = rearrange(timestep, "b v ... -> (b v) ...")
        
        t_emb = self.unet.time_proj(timestep)
        emb = self.unet.time_embedding(t_emb)

        if not self.cfg.disable_rast_cond:
            # Prepare feature pyramid to inject conditioning with rasterized features
            proj_raster_feats = []
            feats = rearrange(rfeatures, "b v c h w -> (b v) c h w")
            feats = self.rasterizer_proj_conv_in(feats)
            pyramid_levels = len(self.rasterizer_proj)
            for idx, layer in enumerate(self.rasterizer_proj):
                if (idx > 0) and (idx < pyramid_levels-1) and (idx % self.residuals_per_level == 0):
                    feats = F.avg_pool2d(feats, kernel_size=2, stride=2)
                feats = layer(feats)
                feats = self.zero_conv_injects[idx](feats)
                proj_raster_feats.append(feats)

        # Set alpha_r weight for conditioning on rasterized features.
        # Not a learnable parameter, just a hack for unconditional generation. 
        alpha_r = 0. if unconditional else 1.

        # Reshape to (b*v, c, h, w) and pre-process (conv_in).
        hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")
        hidden_states = self.unet.conv_in(hidden_states)

        # U-Net downsample blocks.
        down_block_res_samples = (hidden_states,)

        for l, downsample_block in enumerate(self.unet.down_blocks):
            for i, resnet in enumerate(downsample_block.resnets):
                
                hidden_states = resnet(hidden_states, emb)

                if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
    
                    if cond_state is None:
                        cond = torch.zeros(b*num_views, 1, 1024).cuda()
                    else:
                        cond = rearrange(cond_state, "b v d -> (b v) d")
                        cond = cond[:, None, :]
                
                    if isinstance(downsample_block.attentions[i], SpatialTransformer3D):

                        hidden_states = rearrange(hidden_states, "(b v) ... -> b v ...", v=num_views)
                        hidden_states = downsample_block.attentions[i](
                            hidden_states, 
                            extr=extrinsics,
                            intr=intrinsics,
                            plucker_cache=None if not self.cfg.temporal_conditioning else plucker_rays,
                            context=cond
                        )
                        hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")
                    else:
                        hidden_states = downsample_block.attentions[i](
                            hidden_states, encoder_hidden_states=cond
                        ).sample

                down_block_res_samples += (hidden_states,)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        if not self.cfg.disable_rast_cond:
            # Inject rast conditioning via "additional" residuals, similarly to a ControlNet
            new_down_block_res_samples = ()
            for down_block_res_sample, feature_res_sample in zip(down_block_res_samples, proj_raster_feats[:-1]):
                down_block_res_sample = down_block_res_sample + alpha_r * feature_res_sample
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # U-Net mid block.
        hidden_states = self.unet.mid_block.resnets[0](hidden_states, emb)

        for i, (attn, resnet) in enumerate(zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:])):
    
            if cond_state is None:
                cond = torch.zeros(b*num_views, 1, 1024).cuda()
            else:
                cond = rearrange(cond_state, "b v d -> (b v) d")
                cond = cond[:, None, :]

            hidden_states = rearrange(hidden_states, "(b v) ... -> b v ...", v=num_views)
            hidden_states = attn(
                hidden_states,
                extr=extrinsics,
                intr=intrinsics,
                plucker_cache=None if not self.cfg.temporal_conditioning else plucker_rays,
                context=cond)
            hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")
            hidden_states = resnet(hidden_states, emb)

        if not self.cfg.disable_rast_cond:
            # inject rast conditioning
            hidden_states = hidden_states + alpha_r * proj_raster_feats[-1]

        # U-Net upsample blocks.
        for l, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            for i, resnet in enumerate(upsample_block.resnets):
                res_hidden_states = res_samples[-1]
                res_samples = res_samples[:-1]

                hidden_states = torch.cat((hidden_states, res_hidden_states), dim=1)
                hidden_states = resnet(hidden_states, emb)

                if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                    
                    if cond_state is None:
                        cond = torch.zeros(b*num_views, 1, 1024).cuda()
                    else:
                        cond = rearrange(cond_state, "b v d -> (b v) d")
                        cond = cond[:, None, :]

                    if isinstance(upsample_block.attentions[i], SpatialTransformer3D):
                        
                        hidden_states = rearrange(hidden_states, "(b v) ... -> b v ...", v=num_views)
                        hidden_states = upsample_block.attentions[i](
                            hidden_states, 
                            extr=extrinsics,
                            intr=intrinsics,
                            plucker_cache=None if not self.cfg.temporal_conditioning else plucker_rays,
                            context=cond
                        )
                        hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")
                    else:
                        hidden_states = upsample_block.attentions[i](
                            hidden_states, encoder_hidden_states=cond
                        ).sample

            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # Post-process (conv_out) and reshape to (b, v, c, h, w).
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b v) ... -> b v ...', v=num_views)

        return sample
