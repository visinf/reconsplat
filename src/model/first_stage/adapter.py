import torch
from typing import Union, Literal
from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from .discriminator import DiscriminatorPatchGan, DiscriminatorPatchGanCfg
from ..autoencoder import AutoencoderKL, AutoencoderCfg, ColorDecoderCfg, DepthDecoderCfg
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
import copy
from ...misc.nn_module_tools import _count_parameters
import torch.nn as nn

@dataclass
class FirstStageCfg:
    autoencoder_model_name: Literal["kl_f8", "kl_f16", "kl_f32"] = "kl_f8"
    pretrained_autoencoder_path: str | None = None

def _merge_vae_configs(main: AutoencoderCfg, sub: ColorDecoderCfg | DepthDecoderCfg) -> AutoencoderCfg:
    main.up_block_types = sub.up_block_types
    main.block_out_channels = sub.block_out_channels
    # Hack to make the autoencoder builder work, we don't use the encoder.
    if len(sub.up_block_types) < len(main.down_block_types):
        main.down_block_types = main.down_block_types[:len(sub.up_block_types)]

    main.layers_per_block = sub.layers_per_block
    main.latent_channels = sub.latent_channels
    main.skip_connections = sub.skip_connections
    main.skip_extra = sub.skip_extra
    main.skip_zero = sub.skip_zero
    main.pretrained = sub.pretrained
    main.pretrained_path = sub.pretrained_path
    return main

class LearnableGaussianDownsampler(nn.Module):
    """
    Downsample (mu, logvar) from (B,C,H,W) to (B,C,H/f,W/f) with
    moment preservation + learnable residuals.
    """
    def __init__(self, channels: int, factor: int = 8, eps: float = 1e-6):
        super().__init__()
        self.f = factor
        self.eps = eps

        # mu branch: AvgPool (identity) + light refinement
        self.mu_pool = nn.AvgPool2d(factor, factor)
        self.mu_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),  # depthwise
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),  # pointwise fuse
        )
        # init close to identity (small refinement)
        nn.init.zeros_(self.mu_refine[0].weight)
        nn.init.zeros_(self.mu_refine[2].weight)
        nn.init.zeros_(self.mu_refine[0].bias)
        nn.init.zeros_(self.mu_refine[2].bias)

        # std branch: start from mixture variance and learn a positive residual
        # We condition the residual on low-res mu and a pooled local energy map.
        in_v = 2 * channels
        self.var_refine = nn.Sequential(
            nn.Conv2d(in_v, channels, 3, padding=1, groups=channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Softplus()  # ensures residual >= 0
        )
        # small residual at start
        for m in self.var_refine[:-1]:
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, dist: DiagonalGaussianDistribution, mask: torch.Tensor | None = None):
        """
        mu_hr, logvar_hr: [B,C,H,W]
        mask (optional): [B,1,H,W] with {0,1} validity; will re-normalize pooling.
        """
        logvar_hr = dist.logvar
        mu_hr = dist.mean
        var_hr = torch.exp(logvar_hr)

        if mask is None:
            # Downsample first and second moments. 
            mu_lr = self.mu_pool(mu_hr)
            second_moment = self.mu_pool(var_hr + mu_hr**2)
        else:
            # Mask-aware pooling.
            k = self.f
            pool = nn.AvgPool2d(k, k)
            w = pool(mask)
            w = torch.clamp(w, min=self.eps)
            mu_lr = pool(mu_hr * mask) / w
            second_moment = pool((var_hr + mu_hr**2) * mask) / w

        # Compute low-res variance. 
        var_lr = torch.clamp(second_moment - mu_lr**2, min=self.eps)

        # Learnable mu refinement: mu_lr = mu_lr + self.mu_refine(mu_lr).
        mu_lr = mu_lr + self.mu_refine(mu_lr)

        # Learnable var refinement: var_lr = var_lr + self.var_refine([mu_lr, energy_lr]).
        energy_lr = torch.sqrt(torch.clamp(var_lr + mu_lr**2, min=self.eps))
        var_residual = self.var_refine(torch.cat([mu_lr, energy_lr], dim=1))
        var_lr = torch.clamp(var_lr + var_residual, min=self.eps)
        logvar_lr = torch.log(var_lr)

        return DiagonalGaussianDistribution(mean=mu_lr, logvar=logvar_lr)
    
# Small adapter for SD2.1 VAE 
class FirstStageAdapter(torch.nn.Module):
    """
    Adapts diffusers' AutoencoderKL to have:
      - encode(x) -> Tensor (z)
      - decode(z) -> Tensor (x_recon)
    Assumes inputs in [-1, 1] range, outputs in [-1, 1] range.
    """
    def __init__(
            self, 
            vae_cfg: AutoencoderCfg,
            custom_color_decoder_cfg: ColorDecoderCfg,
            custom_depth_decoder_cfg: DepthDecoderCfg,
            discriminator_cfg: DiscriminatorPatchGanCfg | None = None, 
            mode: Literal["train", "test"] = "train",
            stage: Literal['vae', 'diffusion'] = 'vae',
            lora_rank: int = 8,
            lora_alpha: int = 8,
        ):
        super().__init__()
        self.stage = stage
        self.mode = mode

        self.vae_cfg = vae_cfg
        self.app_down = LearnableGaussianDownsampler(channels=4)
        self.geom_down = LearnableGaussianDownsampler(channels=4)

        app_vae_cfg = copy.copy(vae_cfg)
        app_vae_cfg = _merge_vae_configs(app_vae_cfg, custom_color_decoder_cfg)
        geom_vae_cfg = copy.copy(vae_cfg)
        geom_vae_cfg = _merge_vae_configs(geom_vae_cfg, custom_depth_decoder_cfg)

        app_vae_cfg.video_decoder = False
        geom_vae_cfg.video_decoder = False
        
        self.app_vae = AutoencoderKL(app_vae_cfg)
        self.geom_vae = AutoencoderKL(geom_vae_cfg, d_skip_extra=custom_depth_decoder_cfg.d_skip_extra)

        self.app_vae_cfg = app_vae_cfg
        self.geom_vae_cfg = geom_vae_cfg
        self.fine_tune_depth_decoder = custom_depth_decoder_cfg.fine_tune_depth_decoder

        # Enable LoRA-tuning for the VAE color decoder.
        gen_targets = self.find_lora_targets(self.app_vae.model.decoder)

        self.app_vae.model.decoder = get_peft_model(
            self.app_vae.model.decoder,
            peft_config=LoraConfig(
                target_modules=gen_targets,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.1
            )
        )

        if self.fine_tune_depth_decoder:
            # Enable LoRA-tuning for the VAE depth decoder.
            depth_gen_targets = self.find_lora_targets(self.geom_vae.model.decoder)

            self.geom_vae.model.decoder = get_peft_model(
                self.geom_vae.model.decoder,
                peft_config=LoraConfig(
                    target_modules=depth_gen_targets,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1
                )
            )

        if mode == 'train':
            self.app_vae.train()
            self.geom_vae.train()
            self.app_down.train()
            self.geom_down.train()

            self.gan_discriminator = DiscriminatorPatchGan(discriminator_cfg) \
                    if discriminator_cfg is not None else None

            if self.gan_discriminator is not None:
                for p in self.gan_discriminator.parameters():
                    p.requires_grad = True

            print(f'[INFO] Fine-tuning appearance VAE decoder with LoRA.')
            self.app_vae.model.decoder.print_trainable_parameters()
            if self.fine_tune_depth_decoder:
                print(f'[INFO] Fine-tuning geometry VAE decoder with LoRA.')
                self.geom_vae.model.decoder.print_trainable_parameters()

        else:
            self.app_vae.eval()
            self.geom_vae.eval()
            self.app_down.eval()
            self.geom_down.eval()

            self.gan_discriminator = None

        if not self.fine_tune_depth_decoder:
            for p in self.geom_vae.model.decoder.parameters():
                p.requires_grad = False
            for p in self.geom_vae.model.post_quant_conv.parameters():
                p.requires_grad = False
            for p in self.geom_down.parameters():
                p.requires_grad = False

        # NOTE: this code below does not count params for skip connections and post_quant_conv layers.
        print(f'[first-stage] Color Decoder:')
        _count_parameters(self.app_vae.model.decoder, verbose=True)
        print(f'[first-stage] Geometry Decoder:')
        _count_parameters(self.geom_vae.model.decoder, verbose=True)

        if stage == 'vae':
            self.vanilla_vae = None
        else:   
            # For diffusion, also load frozen, pre-trained VAE.
            self.vanilla_vae = AutoencoderKL(vae_cfg)
            for p in self.vanilla_vae.parameters():
                p.requires_grad = False

            self.vanilla_vae.eval()
            self.app_down.eval()
            self.geom_down.eval()

    def find_lora_targets(self, model: torch.nn.Module):  
        names = []
        for name, submod in model.named_modules():
            if isinstance(submod, (torch.nn.Conv2d, torch.nn.Linear)):
                names.append(name)
        return names

    def encode(self, x: torch.Tensor, return_dist: bool = False) -> Union[torch.Tensor, DiagonalGaussianDistribution]:
        # We keep all the encoders frozen, so we can get GT latents with any of them.
        # x: (B,3,H,W) in [-1,1]
        if self.stage == 'vae':
            latent_dist = self.app_vae.vae_encode(x)        # returns a DiagonalGaussianDistribution    
        else:
            latent_dist = self.vanilla_vae.vae_encode(x)

        if return_dist:
            return DiagonalGaussianDistribution(mean=latent_dist.mean, logvar=latent_dist.logvar)
        
        z = latent_dist.sample()            # (B,4,H/8,W/8)
        return z
    
    def downsample_latent_dist(self, dist: DiagonalGaussianDistribution, router: Literal['appearance', 'geometry']) -> DiagonalGaussianDistribution:
        assert router in ['appearance', 'geometry']
        # x: (B,3,H,W) in [-1,1]
        if router == 'appearance':
            latent_dist = self.app_down(dist)   
        else:
            latent_dist = self.geom_down(dist)
        return latent_dist

    def decode(self, z: torch.Tensor, skip_z: torch.Tensor | None = None, router: Literal['appearance', 'geometry', 'vanilla'] = 'appearance', num_frames: int | None = None, image_set: bool = False, **_) -> torch.Tensor:
        assert router in ['appearance', 'geometry', 'vanilla']
        # z: (B,4,H/8,W/8)
        if router == 'appearance':
            dec = self.app_vae.vae_decode(z, skip_z=skip_z, return_dict=True)  # DecoderOutput
        elif router == 'geometry':
            dec = self.geom_vae.vae_decode(z, skip_z=skip_z, return_dict=True) 
        else:
            assert self.vanilla_vae is not None
            dec = self.vanilla_vae.vae_decode(z=z, skip_z=skip_z, return_dict=True, image_set=image_set, num_frames=num_frames)
        
        x = dec.sample # (B,3,H,W) 
        return x

    def last_layer_weights(self) -> list:
        '''
        Extract last layer (decoder.conv_out) weights of the appearance decoder to approximate gradient magnitude 
        of L_rec and L_gen that we use to compute the adaptive weight for the generator loss.
        Extract only trainable params for compatibility with LoRAs.
        '''
        G_L_trainables = [p for p in self.app_vae.model.decoder.conv_out.parameters() if p.requires_grad]
        return G_L_trainables
