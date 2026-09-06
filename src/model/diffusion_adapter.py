from dataclasses import dataclass
from typing import Literal, Tuple, Optional
from jaxtyping import Float
import torch
import torch.nn as nn
from .first_stage.adapter import FirstStageAdapter
from .decoder.decoder import DecoderOutput
from ..dataset.types import BatchedExample
from einops import rearrange, repeat
import numpy as np
from ..misc.camera_utils import absolute_to_relative_camera
import torch.nn.functional as F
from .scheduler import SchedulerCfg, get_scheduler, shift_scheduler_logsnr
from .denoiser import DenoiserCfg, get_denoiser
from .denoiser.guidance.standard import ConstantGuider
from tqdm import tqdm
from ..geometry.projection import get_world_rays, sample_image_grid
from ..misc.depth_io import normalize_depth_quantile, denormalize_depth_quantile
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchvision.utils import save_image
from .denoiser.util import multi_res_noise_like
import math
from ..misc.types import DiffuserOutput
from ..misc.camera_utils import denormalize_K
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.models.autoencoders import AutoencoderKLTemporalDecoder
from .denoiser.clip_conditioner import FrozenOpenCLIPImageEmbedder

@dataclass
class DiffusionAdapterCfg:
    """Config for DiffusionAdapter: denoiser architecture, training scheduler, and sampling options."""
    denoiser: DenoiserCfg
    train_scheduler: SchedulerCfg
    cfg_train: bool=True                                    # Train a conditional and unconditional denoiser.
    use_cfg: bool = False                                   # Use classifier-free guidance when sampling. use_cfg=false is equivalent to a scale of 1.0.
    cfg_scale: float = 3.0
    in_channels: int = 8                                    # Input U-Net channels.
    out_channels: int = 8                                   # Output U-Net channels.
    magic_number: float = 0.18215                           # For scaling VAE latents before diffusion.
    train_active_step: int = 0
    load_ema_weights: Optional[bool] = None                 # None = decide from cfg.mode in main.py (True for test/eval, False for training). 
    replace_conv_in_out: bool = False                       # Used when training a new checkpoint.
    upsample_latents: bool = False
    downsample_preds: bool = False
    disable_depth_modeling: bool = False                    # Ablation.
    apply_multi_res_noise: bool = False
    annealed_mr_noise: bool = False
    mr_noise_strength: float = 0.9
    mr_noise_downscale_strategy: str = "original"
    sample_with_mr_noise: bool = False
    decode_video: bool = False                              # Whether to decode a video or a set of images.
    gaussian_mean: float = 0.0                              # Only used for sampling.
    gaussian_variance: float = 1.0                          # Only used for sampling.
    enable_clip_conditioner_cross_attn: bool = False        # Additional exp./ablation.
    sampling_scheduler: str = "dpm++"
    variational_rast_cond: bool = True                      # Sample the rasterized conditioning during training; always use the mode at inference.
    
def zero_module(m: nn.Module):
    """Zero-init all parameters of a module in place, and return it."""
    for p in m.parameters():
        nn.init.zeros_(p)
    return m

def inflate_conv_(new_w: torch.Tensor, old_w: torch.Tensor):
    """Zero new_w in-place, then copy overlapping slice from old_w."""
    with torch.no_grad():
        new_w.zero_()
        cout = min(new_w.shape[0], old_w.shape[0])
        cin  = min(new_w.shape[1], old_w.shape[1])
        kh   = min(new_w.shape[2], old_w.shape[2])
        kw   = min(new_w.shape[3], old_w.shape[3])
        new_w[:cout, :cin, :kh, :kw].copy_(old_w[:cout, :cin, :kh, :kw])

def inflate_bias_(new_b: torch.Tensor, old_b: torch.Tensor):
    """Zero new_b in-place, then copy the overlapping slice from old_b."""
    with torch.no_grad():
        new_b.zero_()
        n = min(new_b.shape[0], old_b.shape[0])
        new_b[:n].copy_(old_b[:n])

class DiffusionAdapter(nn.Module):
    """Multi-view latent diffusion over color and depth, conditioned on rasterized Gaussian
    features. `forward` computes the training loss inputs; `sample` runs inference-time denoising."""

    def __init__(
        self,
        cfg: DiffusionAdapterCfg,
        autoencoder: FirstStageAdapter,
        image_height: int,
        image_width: int,
        mode: Literal["train", "test"] = "train",
        load_from_ckpt: str | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        replace_conv_in_out = cfg.replace_conv_in_out and mode == "train"
        
        assert cfg.sampling_scheduler in ["ddim", "dpm++"]

        self.train_scheduler = get_scheduler(cfg.train_scheduler)

        if cfg.train_scheduler.shift_schedule:
            if cfg.train_scheduler.shift_by_number_of_views:
                self.train_scheduler = shift_scheduler_logsnr(
                    self.train_scheduler, 
                    N=4, 
                    direction="more_noise"
                ) 
            else:
                self.train_scheduler = shift_scheduler_logsnr(
                    self.train_scheduler,
                    N=4, 
                    shift_by_number_of_views=False,
                    shift_lambda=cfg.train_scheduler.shift_lambda
                )

        # Use DDIM or DPM-Solver++ for sampling. Other samplers are not supported yet in the code.
        if cfg.sampling_scheduler == "ddim":
            self.infer_scheduler_color = DDIMScheduler.from_config(
                self.train_scheduler.config
            )
            self.infer_scheduler_depth = DDIMScheduler.from_config(
                self.train_scheduler.config
            )
        else:
            self.infer_scheduler_color = DPMSolverMultistepScheduler.from_config(
                self.train_scheduler.config
            )
            self.infer_scheduler_depth = DPMSolverMultistepScheduler.from_config(
                self.train_scheduler.config
            )

            if cfg.train_scheduler.shift_schedule: 
                self.infer_scheduler_color.betas = self.train_scheduler.betas
                self.infer_scheduler_color.alphas = self.train_scheduler.alphas
                self.infer_scheduler_color.alphas_cumprod = self.train_scheduler.alphas_cumprod

                self.infer_scheduler_depth.betas = self.train_scheduler.betas
                self.infer_scheduler_depth.alphas = self.train_scheduler.alphas
                self.infer_scheduler_depth.alphas_cumprod = self.train_scheduler.alphas_cumprod
        
        self.autoencoder = autoencoder

        print(f'[diffusion_adapter] using scheduler {cfg.train_scheduler.name} for training.')
        print(f'[diffusion_adapter] using scheduler {cfg.sampling_scheduler} for sampling, '
              f'with {cfg.train_scheduler.num_inference_steps} inference steps.')

        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.image_height = image_height
        self.image_width = image_width

        if self.cfg.disable_depth_modeling:
            self.in_channels = self.in_channels // 2
            self.out_channels = self.out_channels // 2
        
        self.decode_video = self.cfg.decode_video
        self.denoiser = get_denoiser(cfg.denoiser, self.in_channels, self.out_channels, self.image_height, self.image_width, upsample_latents=self.cfg.upsample_latents)

        if self.cfg.enable_clip_conditioner_cross_attn:
            self.clip_conditioner = FrozenOpenCLIPImageEmbedder()
        
        if load_from_ckpt:
            print(f'[diffusion_adapter] Loading ckpt from {load_from_ckpt}')
            ckpt = torch.load(load_from_ckpt, map_location="cpu")
            state_dict = ckpt['state_dict'] 

            prefix = "diffuser.denoiser."

            if replace_conv_in_out:
                # keep a copy of the raw conv_*
                old_conv_in_w  = state_dict.get(f'{prefix}unet.conv_in.weight',  None)
                old_conv_in_b  = state_dict.get(f'{prefix}unet.conv_in.bias',    None)
                old_conv_out_w = state_dict.get(f'{prefix}unet.conv_out.weight', None)
                old_conv_out_b = state_dict.get(f'{prefix}unet.conv_out.bias',   None)
                drop = {
                    "unet.conv_in.weight",
                    "unet.conv_in.bias",
                    "unet.conv_out.weight",
                    "unet.conv_out.bias",
                }

            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if k.startswith(prefix) 
            }
            consume_prefix_in_state_dict_if_present(filtered_state_dict, prefix=prefix)

            if replace_conv_in_out:
                filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if k not in drop}

            missing, unexpected = self.denoiser.load_state_dict(filtered_state_dict, strict=False)

            print("[diffusion_adapter] Loaded MV-LDM checkpoint for denoiser.")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

            if replace_conv_in_out:
                # Inflate/initialize conv_in/out from the pre-trained weights
                unet = self.denoiser.unet
                with torch.no_grad():
                    if old_conv_in_w is not None:
                        inflate_conv_(unet.conv_in.weight, old_conv_in_w.to(unet.conv_in.weight.dtype))
                    else:
                        # fallback: pure zero init if ckpt didn't have it
                        zero_module(unet.conv_in)

                    if old_conv_in_b is not None and hasattr(unet.conv_in, 'bias') and unet.conv_in.bias is not None:
                        inflate_bias_(unet.conv_in.bias, old_conv_in_b.to(unet.conv_in.bias.dtype))
                    elif hasattr(unet.conv_in, 'bias') and unet.conv_in.bias is not None:
                        unet.conv_in.bias.zero_()

                    if old_conv_out_w is not None:
                        inflate_conv_(unet.conv_out.weight, old_conv_out_w.to(unet.conv_out.weight.dtype))
                    else:
                        zero_module(unet.conv_out)

                    if old_conv_out_b is not None and hasattr(unet.conv_out, 'bias') and unet.conv_out.bias is not None:
                        inflate_bias_(unet.conv_out.bias, old_conv_out_b.to(unet.conv_out.bias.dtype))
                    elif hasattr(unet.conv_out, 'bias') and unet.conv_out.bias is not None:
                        unet.conv_out.bias.zero_()

                    print(f"[inflate] conv_in: new {tuple(unet.conv_in.weight.shape)}, old {None if old_conv_in_w is None else tuple(old_conv_in_w.shape)}")
                    print(f"[inflate] conv_out: new {tuple(unet.conv_out.weight.shape)}, old {None if old_conv_out_w is None else tuple(old_conv_out_w.shape)}")

            if self.cfg.load_ema_weights:
                if "ema_state_dict" in ckpt:
                    # EMACallback saves a plain deepcopy of the denoiser, so no prefix to strip here.
                    missing, unexpected = self.denoiser.load_state_dict(ckpt["ema_state_dict"], strict=False)
                    print(f'[diffusion_adapter] Loaded EMA weights for denoiser from {load_from_ckpt}')
                    print("Missing keys:", missing)
                    print("Unexpected keys:", unexpected)
                else:
                    print(f'[diffusion_adapter] cfg.load_ema_weights=True but no "ema_state_dict" found in '
                          f'{load_from_ckpt}; keeping the non-EMA denoiser weights.')

        # Setup CFG guider.
        if self.cfg.use_cfg:
            self.guider = ConstantGuider(scale=self.cfg.cfg_scale)

        print(f'[diffusion_adapter] Training with cfg_train enabled: {self.cfg.cfg_train}')

    def set_inference_timesteps(self, num: Optional[int]=None):
        """
            Args:
                num (Optional[int]): Override the number of inference steps. Default to None.
        """
        num_inference_timesteps = self.cfg.train_scheduler.num_inference_steps if num is None else num
        self.infer_scheduler_color.set_timesteps(num_inference_timesteps) 
        self.infer_scheduler_depth.set_timesteps(num_inference_timesteps)   
        
    def generate_image_rays(
        self,
        images: Float[torch.Tensor, "batch view channel height width"],
        extrinsics: Float[torch.Tensor, "batch view 4 4"],
        intrinsics: Float[torch.Tensor, "batch view 3 3"],
        resolution: Optional[Tuple[int, int]] = None
    ) -> tuple[
        Float[torch.Tensor, "batch view ray 2"],  # xy
        Float[torch.Tensor, "batch view ray 3"],  # origins
        Float[torch.Tensor, "batch view ray 3"],  # directions
    ]:
        """
        Compute ray origins and directions on a grid with shape (height, width). 
        """
        if resolution:
            b, v = images.shape[:2]
            h, w = resolution
        else:
            b, v, _, h, w = images.shape

        device, dtype = extrinsics.device, extrinsics.dtype
        xy, _ = sample_image_grid((h, w), device=device, dtype=dtype)
        origins, directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            rearrange(extrinsics, "b v i j -> b v () i j"),
            rearrange(intrinsics, "b v i j -> b v () i j"),
        )
        return repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), origins, directions

    def ray_encode(
            self,
            context_extrinsics, target_extrinsics,
            context_intrinsics, target_intrinsics,
            context_latents, target_latents, block_res=None):
        """Plucker ray encodings for context+target views. If block_res is given, returns a dict
        of {resolution: encodings} computed at each of those U-Net block resolutions instead of a
        single tensor."""

        if not block_res:
            # Compute Plucker ray encodings at the original resolution.
            b, *_, hl, wl = context_latents.shape
            _, origins_context, directions_context = self.generate_image_rays(context_latents, context_extrinsics, context_intrinsics)
            _, origins_target, directions_target = self.generate_image_rays(target_latents, target_extrinsics, target_intrinsics)
            origins = torch.concat([origins_context, origins_target], dim=1)
            directions = torch.concat([directions_context, directions_target], dim=1)
            origins = torch.cross(origins, directions, dim=-1)
            ray_encodings = torch.concat([origins, directions], dim=-1)   
            ray_encodings = rearrange(ray_encodings, "b v (h w) c -> b v c h w", h=hl, w=wl)
        else:
            ray_dict = {}
            # Pre-compute Plucker ray encodings at different (U-Net) block resolutions.
            b = context_latents.shape[0]
            for hl, wl in block_res:
                _, origins_context, directions_context = self.generate_image_rays(context_latents, context_extrinsics, context_intrinsics, resolution=(hl, wl))
                _, origins_target, directions_target = self.generate_image_rays(target_latents, target_extrinsics, target_intrinsics, resolution=(hl, wl))
                origins = torch.concat([origins_context, origins_target], dim=1)
                directions = torch.concat([directions_context, directions_target], dim=1)
                origins = torch.cross(origins, directions, dim=-1)
                ray_encodings = torch.concat([origins, directions], dim=-1)   
                ray_encodings = rearrange(ray_encodings, "b v (h w) c -> b v c h w", h=hl, w=wl)
                ray_dict[(hl, wl)] = ray_encodings
            return ray_dict
        
        return ray_encodings

    def first_stage_encode(
            self,
            inputs,
            router="appearance",
            normalize=True,
            chunk_size=14,
            downscale_factor=8,
        ):
        """Encode images (router='appearance') or depth (router='geometry') into VAE latents, one
        chunk of frames at a time to bound memory. For depth, also returns a downsampled valid mask."""
        assert router in ['appearance', 'geometry'], "router must be either 'appearance' or 'geometry'"
        b, v, c, h, w = inputs.shape
        inputs = rearrange(inputs, "b v c h w -> (b v) c h w ")

        if normalize:
            if router == 'appearance':
                inputs = inputs * 2.0 - 1.0
            else:
                # Normalize depth to range [-1,1] based on per-scene statistics.
                inputs, valid_mask, (dmin, dmax) = normalize_depth_quantile(inputs, B=b, V=v) 
                inputs = repeat(inputs, "b 1 h w -> b c h w", c=3)
 
        num_frames = inputs.shape[0]
        decoded_chunks = []
        with torch.no_grad():
            for start in range(0, num_frames, chunk_size):
                end = start + chunk_size
                # [chunk_size, c, h, w]
                latents_chunk = inputs[start:end]
                decoded_chunk = self.autoencoder.encode(latents_chunk)
                decoded_chunks.append(decoded_chunk)
        
        latents = torch.cat(decoded_chunks, dim=0) * self.cfg.magic_number
        latents = rearrange(latents, "(b v) c h w -> b v c h w", b=b)

        if router == 'geometry':
            # Downsample valid depth mask 
            invalid_mask = ~valid_mask
            valid_mask_down = ~torch.max_pool2d(
                invalid_mask.float(), downscale_factor, downscale_factor
            ).bool()
            return latents, valid_mask_down

        return latents

    def last_stage_decode(
        self, 
        latents, 
        router="appearance", 
        normalization="image", 
        depth_func="first_channel", # defines how to get depth from the decoded image
        near=None, 
        far=None, 
        chunk_size=14, 
        image_set=False # whether to treat the batch samples as a set of (unordered) images or a video (ordered) sequence.
        ):
        """Decode VAE latents back to images or depth, one chunk of frames at a time. normalization
        controls how the decoded values are mapped back (image: [0,1]; 
        depth: simply return the raw first channel for later post-processing."""

        # What kind on normalization we need to apply (reverse) to the decoded image.
        assert normalization in ["image", "depth", None]
        # How to compute depth from the decoded image.
        assert depth_func in ["first_channel", "average"]

        b, v, c, h, w = latents.shape

        latents = (1 / self.cfg.magic_number) * latents    

        # If we do not have a TemporalDecoder, always treat the incoming latents as an image_set.
        # Note that the TemporalDecoder (SVD VAE Decoder) can also be used for image sets. 
        if hasattr(self.autoencoder, 'vanilla_vae') and router == 'vanilla':
            if not isinstance(self.autoencoder.vanilla_vae.model, AutoencoderKLTemporalDecoder):
                image_set = True
        else:
            image_set = True

        if image_set:
            latents = rearrange(latents, "b v c h w -> (b v) c h w ")
            total_frames = latents.shape[0]
            decoded_chunks = []
            with torch.no_grad():
                for start in range(0, total_frames, chunk_size):
                    end = start + chunk_size
                    # [chunk_size, c, h, w]
                    latents_chunk = latents[start:end]
                    decoded_chunk = self.autoencoder.decode(latents_chunk, router=router, image_set=True)
                    decoded_chunks.append(decoded_chunk)
        else:
            with torch.no_grad():
                decoded_chunks = []
                # Decode each batch scene independently.
                for i in range(latents.shape[0]):
                    latents_i = latents[i]
                    total_frames_per_scene = latents_i.shape[0]
                    decoded_chunks_per_scene = []
                    chunk_size_per_scene = min(total_frames_per_scene, chunk_size)
                    for start in range(0, total_frames_per_scene, chunk_size_per_scene):
                        end = start + chunk_size_per_scene
                        # [chunk_size, c, h, w]
                        latents_chunk = latents_i[start:end]
                        decoded_chunk = self.autoencoder.decode(latents_chunk, router=router, image_set=False, num_frames=latents_chunk.shape[0])
                        decoded_chunks_per_scene.append(decoded_chunk)
                    scene = torch.cat(decoded_chunks_per_scene, dim=0)
                    decoded_chunks.append(scene)
                
        image = torch.cat(decoded_chunks, dim=0)
        image = rearrange(image, "(b v) c h w -> b v c h w", b=b)

        if normalization == 'image':
            image = (image / 2 + 0.5).clamp(0, 1)
        elif normalization == 'depth':
            assert near is not None and far is not None, "near and far must be provided for depth denormalization"
            if depth_func == 'first_channel':
                depth = image[:, :, 0]
            else:
                depth = image.mean(dim=2, keepdim=False)

            image = denormalize_depth_quantile(depth)
        else:
            assert router in ["geometry", "vanilla"]
            if depth_func == 'first_channel':
                image = image[:, :, 0]
            else:
                image = image.mean(dim=2, keepdim=False)
            
        return image

    def forward(
        self,
        batch: BatchedExample,
        rasterized: DecoderOutput,
    ) -> DiffuserOutput:
        """Training step: encodes context/target color+depth to latents, noises the target
        latents, computes model predictions (velocities), and returns predicted and target velocities for the loss."""

        # Context size depends on the dataset and on the view sampler, sometimes (DL3DV).
        context_image = batch["context"]["image"]
        target_image = batch["target"]["image"]
        target_alpha_masks = rasterized.mask.unsqueeze(2)

        b, v_c, c, h, w = context_image.shape
        b, v_t, c, h, w = target_image.shape

        # Convert from absolute to relative extrinsics.
        concat_extr = torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1)
        rel_index = torch.randint(v_c, v_c+v_t, size=(1,)).item()
        rel_extrinsics = absolute_to_relative_camera(concat_extr, index=rel_index).float()
        
        # Note that PRoPE is invariant to the choice of world frame. The relativization is only needed for conditioning the 3D convs, if present.
        extrinsics = torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1)
        intrinsics = torch.concat([batch["context"]["intrinsics"], batch["target"]["intrinsics"]], dim=1)

        # Convert C2W -> W2C as expected by PRoPE encoding.
        extrinsics = extrinsics.inverse() 
        intrinsics = rearrange(intrinsics, "b v i j -> (b v) i j")
        
        # PRoPE also expects K in pixel units.
        intrinsics = denormalize_K(intrinsics, width=self.image_width, height=self.image_height)
        intrinsics = rearrange(intrinsics, "(b v) i j -> b v i j", v=v_c + v_t)

        target_alpha_masks = rearrange(target_alpha_masks, "b vt c h w -> (b vt) c h w")

        if self.cfg.upsample_latents:
            downsampled_target_alpha_masks = F.interpolate(target_alpha_masks, size=(h//4, w//4), mode="bilinear", align_corners=False)
        else:
            downsampled_target_alpha_masks = F.interpolate(target_alpha_masks, size=(h//8, w//8), mode="bilinear", align_corners=False)

        target_depth = batch["target"]["depth"].unsqueeze(2)

        if self.cfg.upsample_latents:
            # Upsample inputs via bilinear interpolation.
            context_image = rearrange(context_image, 'b vc c h w -> (b vc) c h w')
            target_image = rearrange(target_image, 'b vt c h w -> (b vt) c h w')
            target_depth = rearrange(target_depth, 'b vt c h w -> (b vt) c h w')

            context_image = F.interpolate(context_image, scale_factor=2.0, mode="bilinear", align_corners=False) 
            target_image = F.interpolate(target_image, scale_factor=2.0, mode="bilinear", align_corners=False)
            target_depth = F.interpolate(target_depth, scale_factor=2.0, mode="bilinear", align_corners=False)

            context_image = rearrange(context_image, '(b vc) c h w -> b vc c h w', vc=v_c)
            target_image = rearrange(target_image, '(b vt) c h w -> b vt c h w', vt=v_t)
            target_depth = rearrange(target_depth, '(b vt) c h w -> b vt c h w', vt=v_t)

        downscale_factor = 8 if not self.cfg.upsample_latents else 4
        block_res = [
            (h // downscale_factor, w // downscale_factor),
            (h // (downscale_factor*2), w // (downscale_factor*2)),
            (h // (downscale_factor*4), w // (downscale_factor*4)),
            (h // (downscale_factor*8), w // (downscale_factor*8))
        ]

        # Prepare GT latents and noise target (color and depth) latents.
        context_gt_image_latents = self.first_stage_encode(context_image, router="appearance")
        context_gt_depth_placeholder = torch.zeros_like(context_gt_image_latents, device=context_gt_image_latents.device)

        target_gt_image_latents = self.first_stage_encode(target_image, router="appearance")
        target_gt_depth_latents, target_depth_valid_mask = self.first_stage_encode(target_depth, router="geometry")

        # Sample target timesteps.
        timestep_target = torch.randint(
            0, 
            self.cfg.train_scheduler.num_train_timesteps, 
            size=(b,), 
            device=target_gt_image_latents.device,
            dtype=torch.long
        ) 

        # Add noise to target views.
        if self.cfg.apply_multi_res_noise:
            strength = self.cfg.mr_noise_strength
            if self.cfg.annealed_mr_noise:
                timesteps = repeat(timestep_target, "b -> b vt", vt=v_t).flatten()
                strength = strength * (timesteps / self.cfg.train_scheduler.num_train_timesteps)

            target_color_noise = multi_res_noise_like(
                rearrange(target_gt_image_latents, 'b vt c h w -> (b vt) c h w'),
                strength=strength,
                downscale_strategy=self.cfg.mr_noise_downscale_strategy,
                generator=None,
                device=target_gt_image_latents.device
            )
            target_depth_noise = multi_res_noise_like(
                rearrange(target_gt_depth_latents, 'b vt c h w -> (b vt) c h w'),
                strength=strength,
                downscale_strategy=self.cfg.mr_noise_downscale_strategy,
                generator=None,
                device=target_gt_depth_latents.device
            )
            target_color_noise = rearrange(target_color_noise, '(b vt) c h w -> b vt c h w', vt=v_t)
            target_depth_noise = rearrange(target_depth_noise, '(b vt) c h w -> b vt c h w', vt=v_t)
        else:
            target_color_noise = torch.randn_like(target_gt_image_latents).to(target_gt_image_latents.device)
            target_depth_noise = torch.randn_like(target_gt_depth_latents).to(target_gt_depth_latents.device)

        noisy_color_target_latents = self.train_scheduler.add_noise(target_gt_image_latents, target_color_noise, timestep_target)
        noisy_depth_target_latents = self.train_scheduler.add_noise(target_gt_depth_latents, target_depth_noise, timestep_target)

        # Prepare context and target inputs to diffusion model.
        # Pre-compute ray_encodings at different block-resolutions.
        ray_encodings = self.ray_encode(
            rel_extrinsics[:, :v_c, ...], rel_extrinsics[:, v_c:, ...],
            batch["context"]["intrinsics"], batch["target"]["intrinsics"], 
            context_gt_image_latents, target_gt_image_latents, 
            block_res=block_res)
        
        target_ray_encodings = {res: ray_encodings[res][:, v_c:, ...] for res in ray_encodings}

        # We fix zero-noise level for context views.
        timestep_context = torch.zeros((b, ), dtype=torch.long).to(context_gt_image_latents.device)        

        with torch.no_grad():
            # Sample the rasterized conditioning during training, with the VAE reparameterization trick.
            target_rast_color_cond = rasterized.color_posterior.sample() if self.cfg.variational_rast_cond else rasterized.color_posterior.mode()
            target_rast_depth_cond = rasterized.depth_posterior.sample() if self.cfg.variational_rast_cond else rasterized.depth_posterior.mode()

            if self.cfg.upsample_latents:  
                target_rast_color_cond = F.interpolate(target_rast_color_cond, scale_factor=2.0, mode="bilinear", align_corners=False) 
                target_rast_depth_cond = F.interpolate(target_rast_depth_cond, scale_factor=2.0, mode="bilinear", align_corners=False)

        if self.cfg.disable_depth_modeling:
            target_rast_cond = target_rast_color_cond
        else:
            target_rast_cond = torch.concat([target_rast_color_cond, target_rast_depth_cond], dim=1)
        target_rast_cond = target_rast_cond * downsampled_target_alpha_masks
        
        target_rast_cond = rearrange(target_rast_cond, "(b vt) c h w -> b vt c h w", vt=v_t)
        context_rast_cond = torch.zeros((b, v_c, *target_rast_cond.shape[2:]), device=target_rast_cond.device)
        rast_cond = torch.concat([context_rast_cond, target_rast_cond], dim=1)

        if self.cfg.disable_depth_modeling:
            target_inputs = noisy_color_target_latents
        else:
            # Enable joint-modeling of appearance and depth.
            target_inputs = torch.concat([noisy_color_target_latents, noisy_depth_target_latents], dim=2)

        unconditional = False
        if self.cfg.cfg_train:
            # Randomly choose to train conditionally or unconditionally.
            unconditional = np.random.choice([False, True], 1, p=[0.90, 0.10]).item()

        if unconditional:
            # For unconditional training, drop the context views and rasterized features. 
            # Rasterized features are dropped by disabling feature mixing in the unet decoder.
            inputs = target_inputs
            timesteps = repeat(timestep_target, "b -> b vt", vt=v_t)
            rast_cond = target_rast_cond
            plucker_rays = target_ray_encodings
            # Get target-only camera parameters for the unconditional pass.
            extrinsics = extrinsics[:, v_c:, ...]
            intrinsics = intrinsics[:, v_c:, ...]
            cond_state = None
        else:
            plucker_rays = ray_encodings
            # No context depth latents, so we zero them.
            context_depth_latents = torch.zeros_like(context_gt_image_latents).to(context_gt_image_latents.device)
            if self.cfg.disable_depth_modeling:
                context_inputs = context_gt_image_latents
            else:
                context_inputs = torch.concat([context_gt_image_latents, context_depth_latents], dim=2)

            # For the conditional case, use both targets and contexts.
            inputs = torch.concat([context_inputs, target_inputs], dim=1) # (b,v_c+v_t,23,h,w) or (b,v_t,15,h,w) if not activate_rast_cond
            timesteps = torch.concat([
                    repeat(timestep_context, "b -> b vc", vc=v_c),
                    repeat(timestep_target, "b -> b vt", vt=v_t) 
                ], 
                dim=1
            )   

            if self.cfg.enable_clip_conditioner_cross_attn:
                context_image = rearrange(context_image, 'b vc c h w -> (b vc) c h w')
                cond_state = self.clip_conditioner(context_image)
                cond_state = rearrange(cond_state, "(b vc) d -> b vc d", vc=v_c) # (B, vc, D)
                cond_state = torch.mean(cond_state, dim=1) # (B, D)
                cond_state = cond_state[:, None, :].expand(-1, v_c+v_t, -1) # (B, vc+vt, D)
            else:
                cond_state = None

        # Denoise.
        pred = self.denoiser.forward(
            latents=inputs, 
            timestep=timesteps,
            cond_state=cond_state,
            rfeatures=rast_cond, 
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            plucker_rays=plucker_rays,
            unconditional=unconditional)
        
        if unconditional:
            pred_out = pred
        else:
            pred_out = pred[:, v_c:, ...]
            timesteps = timesteps[:, v_c:]

        if not self.cfg.disable_depth_modeling:
            pred_color_out, pred_depth_out = torch.chunk(pred_out, 2, dim=2)
        else:
            # in this ablation, the unet only predicts output color.
            pred_color_out = pred_out
            pred_depth_out = None

        if self.train_scheduler.config.prediction_type == "epsilon":
            target_color_out, target_depth_out = target_color_noise, target_depth_noise
        elif self.train_scheduler.config.prediction_type == "sample":
            target_color_out, target_depth_out = target_gt_image_latents, target_gt_depth_latents
        elif self.train_scheduler.config.prediction_type == "v_prediction":
            
            target_gt_image_latents = rearrange(target_gt_image_latents, "b v c h w -> (b v) c h w")
            target_gt_depth_latents = rearrange(target_gt_depth_latents, "b v c h w -> (b v) c h w")

            target_color_noise = rearrange(target_color_noise, "b v c h w -> (b v) c h w")
            target_depth_noise = rearrange(target_depth_noise, "b v c h w -> (b v) c h w")
            timesteps = rearrange(timesteps, "b v -> (b v)")
            target_color_out = self.train_scheduler.get_velocity(
                target_gt_image_latents, target_color_noise, timesteps
            )
            target_depth_out = self.train_scheduler.get_velocity(
                target_gt_depth_latents, target_depth_noise, timesteps
            )
            
            target_color_out = rearrange(target_color_out, "(b v) c h w -> b v c h w", b=b)
            target_depth_out = rearrange(target_depth_out, "(b v) c h w -> b v c h w", b=b)

        target_depth_valid_mask = rearrange(target_depth_valid_mask, "(b v) c h w -> b v c h w", b=b)

        return DiffuserOutput(
            pred_color=pred_color_out.float(),
            gt_color=target_color_out.float(),
            pred_depth=pred_depth_out.float() if pred_depth_out is not None else None,
            gt_depth=target_depth_out.float(),
            valid_mask_depth=target_depth_valid_mask,
        )

    def step(self, model, x_t_color, x_t_depth, ts, context_inputs, target_rast_cond, extrinsics, intrinsics, ray_encodings, target_ray_encodings, cond_state=None):
        """One inference-time denoising step at timestep ts: predicts noise (with classifier-free
        guidance if use_cfg is set) and advances the color/depth schedulers."""
        b, v_c, *_ = context_inputs.shape
        b, v_t, *_ = x_t_color.shape 

        timestep_context = torch.tensor([0], device=x_t_color.device, dtype=torch.long)
        timestep_target = ts.to(torch.long).to(x_t_color.device).unsqueeze(0)
        timestep_target = repeat(timestep_target, "() -> b", b=b)
        timestep_context = repeat(timestep_context, "() -> b", b=b)
        timesteps = torch.concat([
                repeat(timestep_context, "b -> b vc", vc=v_c),
                repeat(timestep_target, "b -> b vt", vt=v_t) 
            ], 
            dim=1
        )

        if x_t_depth is None:
            assert self.cfg.disable_depth_modeling

        x_t_color_scaled = self.infer_scheduler_color.scale_model_input(x_t_color, ts)
        if self.cfg.disable_depth_modeling:
            target_inputs = x_t_color_scaled
        else:
            x_t_depth_scaled = self.infer_scheduler_depth.scale_model_input(x_t_depth, ts)
            target_inputs = torch.concat([x_t_color_scaled, x_t_depth_scaled], dim=2)

        inputs = torch.concat([context_inputs, target_inputs], dim=1)
        
        target_rast_cond = rearrange(target_rast_cond, "(b vt) c h w -> b vt c h w", vt=v_t)
        context_rast_cond = torch.zeros((b, v_c, *target_rast_cond.shape[2:]), device=target_rast_cond.device)
        rast_cond = torch.concat([context_rast_cond, target_rast_cond], dim=1)

        # Conditional Forward Pass
        pred_conditional = model.forward(
            inputs, 
            timesteps, 
            cond_state=cond_state,
            rfeatures=rast_cond,
            extrinsics=extrinsics, 
            intrinsics=intrinsics,
            plucker_rays=ray_encodings,
            unconditional=False)

        if self.cfg.use_cfg:
            inputs = target_inputs
            timesteps = repeat(timestep_target, "b -> b vt", vt=v_t)

            # Get target-only camera parameters for the unconditional pass.
            target_extrinsics = extrinsics[:, v_c:, ...]
            target_intrinsics = intrinsics[:, v_c:, ...]
            plucker_rays = target_ray_encodings
            
            # Unconditional Forward Pass
            pred_unconditional = model.forward(
                inputs,
                timesteps, 
                cond_state=None,
                rfeatures=target_rast_cond,
                extrinsics=target_extrinsics,
                intrinsics=target_intrinsics,
                plucker_rays=target_ray_encodings,
                unconditional=True)

            # Combine uncond and cond predictions according to criteria defined by CFG guider
            guider_inputs = torch.concat([pred_unconditional, pred_conditional[:, v_c:, ...]], dim=0)
            pred_out = self.guider(guider_inputs, sigma=1.0)
            # pred_out = pred_unconditional + self.cfg.cfg_scale * (pred_conditional[:, v_c:, ...] - pred_unconditional) 
        else:
            pred_out = pred_conditional[:, v_c:, ...]

        if not self.cfg.disable_depth_modeling:
            pred_out_color, pred_out_depth = torch.chunk(pred_out, 2, dim=2)
        else:
            pred_out_color = pred_out
            pred_out_depth = None

        # Hack to test with a batch size > 1 - only works if all noisy samples are at the same diffusion timestep. 
        pred_out_color = rearrange(pred_out_color, "b v c h w -> (b v) c h w")
        x_t_color = rearrange(x_t_color, "b v c h w -> (b v) c h w")
        sch_out_color = self.infer_scheduler_color.step(
            pred_out_color, 
            ts,
            x_t_color, 
            return_dict=True
        )

        x_tprev_color = rearrange(sch_out_color.prev_sample, "(b v) c h w -> b v c h w", b=b)

        if pred_out_depth is not None:
            pred_out_depth = rearrange(pred_out_depth, "b v c h w -> (b v) c h w")
            x_t_depth = rearrange(x_t_depth, "b v c h w -> (b v) c h w")
            sch_out_depth = self.infer_scheduler_depth.step(
                pred_out_depth, 
                ts, 
                x_t_depth, 
                return_dict=True
            )
            x_tprev_depth = rearrange(sch_out_depth.prev_sample, "(b v) c h w -> b v c h w", b=b)
        else:
            x_tprev_depth = None

        return (x_tprev_color, x_tprev_depth)

    def sample(
        self,
        batch: BatchedExample,
        rasterized: DecoderOutput,
        return_dict: bool = False,
        color_noise: torch.Tensor | None = None,
        depth_noise: torch.Tensor | None = None,
        decode_video: bool = False,
        depth_normalization: bool = True,
        skip_decode: bool = False,  # if True, return the raw denoised latents instead of decoding them.
    )-> Tuple[torch.Tensor, torch.Tensor] | dict:
        """Inference-time sampling: denoises target color+depth latents from noise for a given number of timesteps,
        then decodes them to images/depth unless skip_decode."""

        b, v_c, c, h, w = batch["context"]["image"].shape
        b, v_t, _, _ = batch["target"]["extrinsics"].shape
        device = batch["context"]["image"].device
        print(f'[diffusion_adapter] Sampling with {self.cfg.sampling_scheduler} and cfg scale = {self.cfg.cfg_scale}')

        # Reset the samplers so this run starts at t=T with a clean step_index (a second sample()
        # call in the same step would otherwise resume mid-schedule).
        self.set_inference_timesteps()

        model = self.denoiser
        model.eval()

        decode_video = self.cfg.decode_video or decode_video
        image_set=False if decode_video else True

        # Convert from absolute to relative extrinsics.
        concat_extr = torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1)
        rel_index = torch.randint(v_c, v_c+v_t, size=(1,)).item()
        rel_extrinsics = absolute_to_relative_camera(concat_extr, index=rel_index).float()

        extrinsics = torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1)
        intrinsics = torch.concat([batch["context"]["intrinsics"], batch["target"]["intrinsics"]], dim=1)
        
        # Convert C2W -> W2C, as expected by PRoPE encoding.
        extrinsics = extrinsics.inverse() 
        intrinsics = rearrange(intrinsics, "b v i j -> (b v) i j")
        
        # PRoPE also expects K in pixel units.
        intrinsics = denormalize_K(intrinsics, width=self.image_width, height=self.image_height) 
        intrinsics = rearrange(intrinsics, "(b v) i j -> b v i j", v=v_c + v_t)

        context_images = batch["context"]["image"]

        if self.cfg.upsample_latents:
            context_images = rearrange(context_images, "b vc c h w -> (b vc) c h w")
            context_images = F.interpolate(context_images, scale_factor=2.0, mode="bilinear", align_corners=False)
            context_images = rearrange(context_images, "(b vc) c h w -> b vc c h w", vc=v_c)

        target_alpha_masks = rasterized.mask.unsqueeze(2)        
        target_alpha_masks = rearrange(target_alpha_masks, "b vt c h w -> (b vt) c h w")

        downscale_factor = 8 if not self.cfg.upsample_latents else 4
        block_res = [
            (h // downscale_factor, w // downscale_factor),
            (h // (downscale_factor*2), w // (downscale_factor*2)),
            (h // (downscale_factor*4), w // (downscale_factor*4)),
            (h // (downscale_factor*8), w // (downscale_factor*8))
        ]

        if self.cfg.upsample_latents:
            downsampled_target_alpha_masks = F.interpolate(target_alpha_masks, size=(h//4, w//4), mode="bilinear", align_corners=False)
        else:
            downsampled_target_alpha_masks = F.interpolate(target_alpha_masks, size=(h//8, w//8), mode="bilinear", align_corners=False)

        with torch.no_grad():
            context_image_latents = self.first_stage_encode(context_images, router='appearance') # (b v c h w)
            
            if self.cfg.enable_clip_conditioner_cross_attn:
                cond_state = self.clip_conditioner(rearrange(context_images, 'b vc c h w -> (b vc) c h w'))
                cond_state = rearrange(cond_state, "(b vc) d -> b vc d", vc=v_c)
                cond_state = torch.mean(cond_state, dim=1) # (B, D)
                cond_state = cond_state[:, None, :].expand(-1, v_c+v_t, -1) # (B, vc+vt, D)
            else:
                cond_state = None

            mu = self.cfg.gaussian_mean
            # Compute the standard deviation of the Gaussian noise based on the configured variance.
            sigma = (self.cfg.gaussian_variance)**0.5

            # Sample init noise for color and depth, if not provided.
            if color_noise is None:
                color_noise = torch.randn((context_image_latents.shape[0], v_t, *context_image_latents.shape[2:])).to(context_image_latents.device)
                if self.cfg.sample_with_mr_noise:
                    print("[diffuser] Applying multi-resolution noise during sampling.")
                    color_noise = multi_res_noise_like(
                        rearrange(color_noise, 'b vt c h w -> (b vt) c h w'),
                        strength=self.cfg.mr_noise_strength,
                        downscale_strategy=self.cfg.mr_noise_downscale_strategy,
                        generator=None,
                        device=context_image_latents.device
                    )
                    color_noise = rearrange(color_noise, '(b vt) c h w -> b vt c h w', vt=v_t)
                
                x_t_color = mu + sigma * color_noise
            else:
                x_t_color = color_noise

            x_t_color *= self.infer_scheduler_color.init_noise_sigma 
            
            if depth_noise is None:

                depth_noise = torch.randn((context_image_latents.shape[0], v_t, *context_image_latents.shape[2:])).to(context_image_latents.device)
                if self.cfg.sample_with_mr_noise:
                    print("[diffuser] Applying multi-resolution noise during sampling.")
                    depth_noise = multi_res_noise_like(
                        rearrange(depth_noise, 'b vt c h w -> (b vt) c h w'),
                        strength=self.cfg.mr_noise_strength,
                        downscale_strategy=self.cfg.mr_noise_downscale_strategy,
                        generator=None,
                        device=context_image_latents.device
                    )
                    depth_noise = rearrange(depth_noise, '(b vt) c h w -> b vt c h w', vt=v_t)

                x_t_depth = mu + sigma * depth_noise
            else:
                x_t_depth = depth_noise

            x_t_depth *= self.infer_scheduler_depth.init_noise_sigma 

            ray_encodings = self.ray_encode(
                rel_extrinsics[:, :v_c, ...], rel_extrinsics[:, v_c:, ...],
                batch["context"]["intrinsics"], batch["target"]["intrinsics"], 
                context_image_latents, x_t_color, 
                block_res=block_res)
            target_ray_encodings = {res: ray_encodings[res][:, v_c:, ...] for res in ray_encodings}

            # Always use the mode (not the sample) for conditioning at inference, for stable/reproducible results.
            target_rast_color_cond = rasterized.color_posterior.mode()
            target_rast_depth_cond = rasterized.depth_posterior.mode()

            if self.cfg.upsample_latents:  
                target_rast_color_cond = F.interpolate(target_rast_color_cond, scale_factor=2.0, mode="bilinear", align_corners=False) 
                target_rast_depth_cond = F.interpolate(target_rast_depth_cond, scale_factor=2.0, mode="bilinear", align_corners=False)

            # Prepare conditioning with rasterized features.
            if self.cfg.disable_depth_modeling:
                target_rast_cond = target_rast_color_cond
            else:
                target_rast_cond = torch.concat([target_rast_color_cond, target_rast_depth_cond], dim=1)

            target_rast_cond = target_rast_cond * downsampled_target_alpha_masks

            # Always "drop" extra conditioning for context views.
            context_depth_latents = torch.zeros_like(context_image_latents).to(context_image_latents.device)
            
            if self.cfg.disable_depth_modeling:
                context_inputs = context_image_latents
            else:
                context_inputs = torch.concat([context_image_latents, context_depth_latents], dim=2)

            for i, ts in enumerate(tqdm(self.infer_scheduler_color.timesteps)):
                x_t_color, x_t_depth = self.step(
                    model,
                    x_t_color,
                    x_t_depth,
                    ts,
                    context_inputs,
                    target_rast_cond,
                    extrinsics,
                    intrinsics,
                    ray_encodings,
                    target_ray_encodings,
                    cond_state=cond_state
                    )

            if skip_decode:
                if return_dict:
                    return {
                        "color_latents_raw": x_t_color,
                        "depth_latents_raw": x_t_depth,
                        "latent_color_sample": target_rast_color_cond,
                        "latent_depth_sample": target_rast_depth_cond,
                    }
                return x_t_color, x_t_depth

            color_denoised = self.last_stage_decode(
                x_t_color, 
                router="vanilla", 
                normalization="image",
                image_set=image_set
            )

            if x_t_depth is not None:
                assert not self.cfg.disable_depth_modeling
                depth_denoised = self.last_stage_decode(
                    x_t_depth, 
                    router="vanilla", 
                    normalization="depth" if depth_normalization else None,
                    near=batch["target"]["near"], 
                    far=batch["target"]["far"],
                    image_set=image_set
                )
            else:
                depth_denoised = None

            if self.cfg.downsample_preds:
                color_denoised = rearrange(color_denoised, "b vt c h w -> (b vt) c h w")
                # Downsample model prediction back to original resolution.
                color_denoised = F.interpolate(
                    color_denoised, 
                    scale_factor = 0.5,
                    mode="bilinear",
                    align_corners=False
                )
                color_denoised = rearrange(color_denoised, "(b vt) c h w -> b vt c h w", vt=v_t)

                if depth_denoised is not None:
                    depth_denoised = depth_denoised.unsqueeze(2)
                    depth_denoised = rearrange(depth_denoised, "b vt 1 h w -> (b vt) 1 h w")
                    depth_denoised = F.interpolate(
                        depth_denoised,
                        scale_factor = 0.5, 
                        mode="bilinear",
                        align_corners=False
                    )
                    depth_denoised = rearrange(depth_denoised, "(b vt) 1 h w -> b vt 1 h w", vt=v_t)
                    depth_denoised = depth_denoised.squeeze(2)

                target_rast_color_cond = F.interpolate(target_rast_color_cond, scale_factor=0.5, mode="bilinear", align_corners=False)
                target_rast_depth_cond = F.interpolate(target_rast_depth_cond, scale_factor=0.5, mode="bilinear", align_corners=False)

        if return_dict:
            return {
                "color_denoised": color_denoised.float(),
                "depth_denoised": depth_denoised.float() if depth_denoised is not None else None,
                "latent_color_sample": target_rast_color_cond,
                "latent_depth_sample": target_rast_depth_cond,
            }

        return color_denoised.float(), depth_denoised.float()

    def active(self, global_step: int):
        return global_step >= self.cfg.train_active_step