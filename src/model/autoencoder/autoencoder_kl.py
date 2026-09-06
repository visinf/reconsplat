from dataclasses import dataclass
import os
from typing import Literal, Optional, Union
import itertools
from diffusers import (
    AutoencoderKL as Model,
)

from diffusers.models.autoencoders import (
    AutoencoderKLTemporalDecoder as ModelTemporalDecoder
)

from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution as LatentDistribution,
)
from diffusers.utils.accelerate_utils import apply_forward_hook
from jaxtyping import Float
import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate

from .autoencoder import Autoencoder
from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ...misc.nn_module_tools import zero_module
from safetensors import safe_open

def load_partial_state(ckpt_path, prefix):
    partial_state = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                partial_state[k[len(prefix):]] = f.get_tensor(k)  # loads only this tensor (memory-mapped)
    print(f"Loaded {len(partial_state)} tensors from {prefix}")
    return partial_state

@dataclass
class ColorDecoderCfg:
    up_block_types: list[str]
    block_out_channels: list[int]
    layers_per_block: int
    latent_channels: int   
    skip_connections: bool = False
    skip_extra: bool = True
    d_skip_extra: int = 0
    skip_zero: bool = True
    pretrained: bool = True
    pretrained_path: str | None = None

@dataclass
class DepthDecoderCfg:
    up_block_types: list[str]
    block_out_channels: list[int]
    layers_per_block: int
    latent_channels: int   
    skip_connections: bool = False
    skip_extra: bool = True
    d_skip_extra: int = 0
    skip_zero: bool = True
    pretrained: bool = True
    pretrained_path: str | None = None
    fine_tune_depth_decoder: bool = True

@dataclass
class AutoencoderKLCfg:
    name: Literal["kl"]
    model: Literal["kl_f8", "kl_f16", "kl_f32"]
    down_block_types: list[str]
    up_block_types: list[str]
    block_out_channels: list[int]
    layers_per_block: int
    latent_channels: int
    skip_connections: bool = False
    skip_extra: bool = True
    skip_zero: bool = True
    video_decoder: bool = False
    video_decoder_path: str = None
    pretrained: bool = True
    pretrained_path: str = None

class AutoencoderKL(Autoencoder[AutoencoderKLCfg]):
    def __init__(
        self, 
        cfg: AutoencoderKLCfg, 
        d_in: int = 3,
        d_skip_extra: int = 0,
        sample_size: int = 32,
    ) -> None:
        super().__init__(cfg)
        if cfg.video_decoder:
            self.model = ModelTemporalDecoder(
                d_in,
                d_in,
                down_block_types=self.cfg.down_block_types,
                block_out_channels=self.cfg.block_out_channels,
                layers_per_block=self.cfg.layers_per_block,
                latent_channels=self.cfg.latent_channels,
                sample_size=sample_size
            )
        else:
            self.model = Model(
                d_in,
                d_in,
                down_block_types=self.cfg.down_block_types,
                up_block_types=self.cfg.up_block_types,
                block_out_channels=self.cfg.block_out_channels,
                layers_per_block=self.cfg.layers_per_block,
                latent_channels=self.cfg.latent_channels,
                sample_size=sample_size
            )
        
        # NOTE: if cfg.video_decoder, load the autoencoder weights from the SVD checkpoints (avoid loading the entire thing to memory).
        if self.cfg.pretrained and self.cfg.pretrained_path is not None:
            if self.cfg.video_decoder:
                # NOTE: Doing this trick rn of using the loader from diffusers, to avoid messing up with the key mapping between SVD and diffusers' components...
                #       But better to do it offline, SVD page can go down...
                from diffusers import StableVideoDiffusionPipeline
                svd = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid",
                    torch_dtype=torch.float32,
                )
                self.model = svd.vae
                # # Load "first stage" model from SVD checkpoint.
                # state_dict = load_partial_state(
                #     ckpt_path="checkpoints/video/svd.safetensors",
                #     prefix = "first_stage_model."
                # )
                # self.model.load_state_dict(state_dict)
            else:
                state_dict = torch.load(os.path.join(self.cfg.pretrained_path, self.cfg.model + ".pt"), map_location="cpu")
                self.model.load_state_dict(state_dict)

        if self.cfg.skip_connections:
            # Add zero convs for high-resolution skip connections
            self.d_skip = self.d_latent
            if self.cfg.skip_extra:
                self.d_skip += d_skip_extra
            skip = nn.Conv2d(self.d_skip, self.cfg.block_out_channels[-1], kernel_size=1)
            if self.cfg.skip_zero:
                skip = zero_module(skip)
            self.skip_convs = nn.ModuleList([skip])
            for dec_d_in in reversed(self.cfg.block_out_channels):
                skip = nn.Conv2d(self.d_skip, dec_d_in, kernel_size=1)
                if self.cfg.skip_zero:
                    skip = zero_module(skip)
                self.skip_convs.append(skip)
        
    def encode(
        self, 
        images: Float[Tensor, "*#batch d_img height width"]
    ) -> DiagonalGaussianDistribution:
        """
        Expects channels to be in range [0, 1]
        Returns distribution with original number of batch dimensions
        """
        batch_dims = images.shape[:-3]
        images = 2 * images - 1         # normalize to [-1, 1]
        images = images.flatten(0, -4)  # make sure to have just one batch dimension
        latent_dist: LatentDistribution = self.vae_encode(images)
        return DiagonalGaussianDistribution(
            mean=latent_dist.mean.reshape(*batch_dims, *latent_dist.mean.shape[1:]),
            logvar=latent_dist.logvar.reshape(*batch_dims, *latent_dist.logvar.shape[1:]),
        )
    
    def vae_encode(
        self, 
        images: Float[Tensor, "#batch d_img height width"]
    ) -> LatentDistribution:
        return self.model.encode(images).latent_dist
    
    def _temporal_decoder_forward(
        self, 
        sample: torch.Tensor,
        image_only_indicator: torch.Tensor,
        num_frames: int = 1,
    ) -> torch.Tensor:
        decoder = self.model.decoder
        
        r"""The forward method of the `Decoder` class."""
        sample = decoder.conv_in(sample)
        upscale_dtype = next(itertools.chain(decoder.up_blocks.parameters(), decoder.up_blocks.buffers())).dtype
        if torch.is_grad_enabled() and decoder.gradient_checkpointing:
            # middle
            sample = decoder._gradient_checkpointing_func(
                decoder.mid_block,
                sample,
                image_only_indicator,
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in decoder.up_blocks:
                sample = decoder._gradient_checkpointing_func(
                    up_block,
                    sample,
                    image_only_indicator,
                )
        else:
            # middle
            sample = decoder.mid_block(sample, image_only_indicator=image_only_indicator)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in decoder.up_blocks:
                sample = up_block(sample, image_only_indicator=image_only_indicator)

        # post-process
        sample = decoder.conv_norm_out(sample)
        sample = decoder.conv_act(sample)
        sample = decoder.conv_out(sample)

        batch_frames, channels, height, width = sample.shape
        batch_size = batch_frames // num_frames
        sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        sample = decoder.time_conv_out(sample)

        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

        return sample

    def _decoder_foward(
        self,
        z: torch.Tensor,
        skip_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""
        decoder = self.model.decoder
        z = decoder.conv_in(z)

        upscale_dtype = next(iter(decoder.up_blocks.parameters())).dtype
        # middle
        z = decoder.mid_block(z)
        z = z.to(upscale_dtype)

        # up
        for i, up_block in enumerate(decoder.up_blocks):
            if self.cfg.skip_connections:
                # Apply skip conv layer
                z = z + self.skip_convs[i](interpolate(
                    skip_z, 
                    size=z.shape[-2:], 
                    mode="bilinear", 
                    align_corners=True
                ))
            z = up_block(z)

        # post-process
        z = decoder.conv_norm_out(z)
        z = decoder.conv_act(z)
        z = decoder.conv_out(z)

        return z

    def _vae_decode(
        self, 
        z: torch.Tensor,
        skip_z: Optional[torch.Tensor] = None,
        num_frames: int | None = None,
        return_dict: bool = True,
        image_set: bool = False
    ) -> Union[DecoderOutput, torch.Tensor]:
        # z = self.model.post_quant_conv(z) 
        # To be compatible with decoders without post_quant_conv. 
        pqc = getattr(self.model, "post_quant_conv", None)
        if callable(pqc):
            z = pqc(z)

        # Check if we have a temporal decoder.
        if isinstance(self.model, ModelTemporalDecoder):
            if num_frames is None:
                assert len(z.shape) == 4
                num_frames = z.shape[0]

            batch_size = z.shape[0] // num_frames
            if image_set:
                # Treat samples as images, not video frames, i.e., disable temporal blocks in the decoder.
                image_only_indicator = torch.ones(batch_size * num_frames, 1, dtype=z.dtype, device=z.device)
                num_frames = 1
            else:
                image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
            
            dec = self._temporal_decoder_forward(z, image_only_indicator=image_only_indicator, num_frames=num_frames)
        else: 
            dec = self._decoder_foward(z, skip_z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def vae_decode(
        self, 
        z: torch.Tensor,
        skip_z: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        image_set: bool = False,
        num_frames: int | None = None
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        decoded = self._vae_decode(z, skip_z, image_set=image_set, num_frames=num_frames).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def decode(
        self, 
        z: Float[Tensor, "*#batch d_latent latent_height latent_width"],
        skip_z: Optional[Float[Tensor, "*#batch d_skip height width"]] = None,
        image_set: bool = False,
        num_frames: int | None = None,
    ) -> Float[Tensor, "*#batch d_img height width"]:
        batch_dims = z.shape[:-3]
        z = z.flatten(0, -4)
        if skip_z is not None:
            skip_z = skip_z.flatten(0, -4)
        sample = self.vae_decode(z, skip_z, image_set=image_set, num_frames=num_frames).sample
        sample = (sample + 1) / 2
        sample = sample.reshape(*batch_dims, *sample.shape[1:])
        return sample

    @property
    def downscale_factor(self) -> int:
        return 2 ** (len(self.cfg.block_out_channels)-1)

    @property
    def d_latent(self) -> int:
        return self.cfg.latent_channels
    
    @property
    def last_layer_weights(self) -> Tensor:
        return self.model.decoder.conv_out.weight

    @property
    def expects_skip(self) -> bool:
        return self.cfg.skip_connections
    
    @property
    def expects_skip_extra(self) -> bool:
        return self.cfg.skip_extra