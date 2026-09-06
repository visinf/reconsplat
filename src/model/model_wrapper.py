from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Any, Dict, runtime_checkable, Tuple, Literal

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import torch.nn.functional as F
import numpy as np
import json
from warnings import warn
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, compute_dists
from ..global_cfg import get_cfg
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, render_error_map, render_depth_map_percentile as render_depth_map
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_pointcloud_projections_from_rgbd
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .first_stage.adapter import FirstStageAdapter
from .first_stage.decoding_utils import stitch_chunks
from .diffusion_adapter import DiffusionAdapter
from ..loss import LossDict
from ..model.decoder.decoder import DecoderOutput
from ..model.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ..misc.depth_io import fit_log_affine_scene, apply_log_affine_scene
from ..misc.variational_downsample import downsample_mu_logvar_mixture
from ..misc.types import DiffuserOutput
import os

@dataclass 
class LRSchedulerCfg:
    warm_up_steps: int
    cosine_lr: bool
    final_div_factor: float
    kwargs: Optional[Dict[str, Any] ] = None

@dataclass
class GeneratorOptimizerCfg:
    lr: float
    scale_lr: bool
    gradient_clip_val: float = 0.5
    kwargs: Optional[Dict[str, Any] ] = None
    lr_scheduler: Optional[LRSchedulerCfg] = None

@dataclass
class DiscriminatorOptimizerCfg:
    lr: float
    scale_lr: bool
    gradient_clip_val: float = 0.5
    kwargs: Optional[Dict[str, Any]] = None
    lr_scheduler: Optional[LRSchedulerCfg] = None

@dataclass
class OptimizerCfg:
    generator: GeneratorOptimizerCfg
    discriminator: Optional[DiscriminatorOptimizerCfg] = None

@dataclass
class TestCfg:
    """Options for mode=test (evaluation/inference), see ModelWrapper.test_step/on_test_end."""
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    fix_noise_for_targets: bool = False
    sample_in_chunks: bool = False
    chunk_size: int = 14
    chunk_overlap: int = 4
    chunk_decode_mode: Literal["blend_then_decode", "save_latents"] = "blend_then_decode" # NOTE: Use only as a fallback for decoding several target views at once on smaller GPUs.
    chunk_decode_fps: int = 20                      # FPS used by src/scripts/decode_latent_chunks.py when chunk_decode_mode=save_latents
    save_intermediate_recons: bool = False          # Also save the rasterized (pre-diffusion) color/depth reconstructions
    
    # To produce qualitative results during validation.
    generate_val_video_wobble: bool = True
    val_video_num_frames: int = 30                  # Number of frames we use to generate wobble trajectories during validation.
    generate_pcd_projections: bool = False          # Whether to visualize point clouds during validation. 
    align_pcd_with_rasterized_depth: bool = True    # Use the rasterized depth to align the predicted depth before unprojecting.

    # Test-time only. 
    fit_pcd_scale: bool = False
    # Depth scale for mapping depth predictions from the "raw" range [-1, 1] to the same scale of camera poses:
    # - "oracle": fit to saved depth pseudo-labels or VGGT depth at context views; requires predicting depth for context views as well.
    # - "rasterized_all_views": fit to 3DGS-rasterized depth over all target views; may degrade under extrapolation.
    # - "rasterized_context_views": fit to predicted depth for context views from the cost-volume encoder.
    pcd_alignment_source: Literal["oracle", "rasterized_all_views", "rasterized_context_views"] = "oracle"

Stage = Literal["vae", "diffusion"]

@dataclass
class TrainCfg:
    """Options for mode=train. stage selects which part of the model is being trained ('vae' or
    'diffusion', see ModelWrapper.__init__)."""
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    stage: Stage

@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass

def _disable_gradient_tracking(module: nn.Module | None):
    """Freeze a module: no gradients, and .eval() so e.g. BatchNorm stops updating its running
    stats. Callers must also add it to self._frozen_modules (see ModelWrapper.train)."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def make_windows(T=50, W=14, O=4, allow_short_last=False):
    """Split a length-T sequence into overlapping windows of size W with overlap O."""
    S = W - O

    if allow_short_last:
        windows = []
        start = 0
        while start < T:
            end = min(start + W, T)
            windows.append(list(range(start, end)))
            if end == T:
                break
            start += S
        return windows

    # Original behavior
    starts = list(range(0, max(T - W + 1, 1), S))
    if starts[-1] != T - W:
        starts.append(T - W)
    windows = [list(range(s, s + W)) for s in starts]
    return windows

class ModelWrapper(LightningModule):
    """PyTorch Lightning module tying together the encoder (Gaussians from context views), decoder
    (rasterization), and optionally a first-stage VAE / diffusion adapter, depending on
    train_cfg.stage ('vae' or 'diffusion'). Handles training, validation, and test/eval."""

    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: LossDict
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    first_stage: FirstStageAdapter | None
    diffuser: DiffusionAdapter | None
    step_tracker: StepTracker | None
    variational_features: bool = True

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        first_stage: FirstStageAdapter | None,
        diffuser: DiffusionAdapter | None,
        losses: LossDict | None,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        # "vae" stage manually alternates generator/discriminator steps, so it needs manual
        # optimization. "diffusion" has a single optimizer, so it uses Lightning's automatic mode
        # (needed for accumulate_grad_batches to work correctly).
        self.automatic_optimization = train_cfg.stage != "vae"
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.first_stage = first_stage
        self.data_shim = get_data_shim(self.encoder)
        self.losses = losses

        self.diffuser = diffuser if self.train_cfg.stage == 'diffusion' else None
        if self.diffuser is not None and self.losses['diff_losses']['diffusion'] is not None:
            # Make configs consistent
            self.diffuser.cfg.train_active_step = self.losses['diff_losses']['diffusion'].cfg.apply_after_step

        print(f'[INFO] Training Stage: {self.train_cfg.stage}')

        # NOTE: By default, the first-stage adapter (VAE generator and patch GAN discriminator)
        #       has gradient tracking enabled for all the params.
        # Depending on the training stage, disable gradient tracking for some blocks.
        if self.train_cfg.stage == 'vae':
            # We do not need the actual VAE encoder at this stage.
            self._frozen_modules = [
                self.first_stage.app_vae.model.encoder,
                self.first_stage.app_vae.model.quant_conv,
                self.first_stage.geom_vae.model.encoder,
                self.first_stage.geom_vae.model.quant_conv,
            ]
        else:
            # For the diffusion stage, disable everything but the diffuser.
            self._frozen_modules = [
                self.encoder,
                self.decoder,
                self.first_stage.app_vae,
                self.first_stage.app_down,
                self.first_stage.geom_vae,
                self.first_stage.geom_down,
            ]
        for module in self._frozen_modules:
            _disable_gradient_tracking(module)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def train(self, mode: bool = True):
        """Like nn.Module.train(), but re-applies .eval() to the frozen submodules afterward, since
        Lightning's own train() call would otherwise put them back in train mode too."""
        super().train(mode)
        for module in self._frozen_modules:
            module.eval()
        return self

    def setup(self, stage: str) -> None:
        """Lightning hook: scales the configured base learning rates by the effective batch size, if scale_lr is True."""
        if stage == "fit":
            assert self.train_cfg.stage != 'vae' or self.trainer.accumulate_grad_batches == 1, \
                "Gradient accumulation is not supported for the 'vae' stage (manual optimization, " \
                "alternating generator/discriminator steps)."
            # assumes one fixed batch_size for all train dataloaders!
            effective_batch_size = self.trainer.accumulate_grad_batches \
                * self.trainer.num_devices \
                * self.trainer.num_nodes \
                * self.trainer.datamodule.data_loader_cfg.train.batch_size
            
            self.generator_lr = effective_batch_size * self.optimizer_cfg.generator.lr \
                if self.optimizer_cfg.generator.scale_lr else self.optimizer_cfg.generator.lr
            if self.optimizer_cfg.discriminator is not None:
                self.discriminator_lr = effective_batch_size * self.optimizer_cfg.discriminator.lr \
                    if self.optimizer_cfg.discriminator.scale_lr else self.optimizer_cfg.discriminator.lr
                
        return super().setup(stage)

    def visualize_latent_fields(
            self, 
            color_posterior: DiagonalGaussianDistribution,
            depth_posterior: DiagonalGaussianDistribution,
            n_images: int,
            sample_mode: bool = False
        ):
        """Decode the first-stage color/depth samples straight to an image for visualization. """
        if sample_mode:
            color_feature = color_posterior.mode()
        else:
            color_feature = color_posterior.sample() if self.variational_features else color_posterior.mode()
        if sample_mode:
            depth_feature = depth_posterior.mode()
        else:
            depth_feature = depth_posterior.sample() if self.variational_features else depth_posterior.mode()

        color_feature = color_feature[:n_images, :3]
        depth_feature = depth_feature[:n_images, :3]

        renderings = [
            add_label(hcat(*color_feature), "Color (Decoded)"),
            add_label(hcat(*depth_feature), "Depth (Decoded)"),
        ]
        comparison = vcat(*renderings)  
        return comparison, color_feature, depth_feature

    def downsample_latent_field(
        self, 
        latent_field: DiagonalGaussianDistribution,
        downsample_factor: int
    ):
        """Downsample (mu, logvar) by matching the first two moments of the
        mixture inside each non-overlapping block."""
        mu_lr, logvar_lr = downsample_mu_logvar_mixture(
            latent_field.mean,
            latent_field.logvar,
            factor=downsample_factor,
            eps=1e-10
        )
        return DiagonalGaussianDistribution(mean=mu_lr, logvar=logvar_lr)

    def decode_rasterized_latent(
            self, 
            batch: BatchedExample,
            output: DecoderOutput, 
            h_latent: int, 
            w_latent: int,
            sample_depth_mode: bool = False,
            return_high_res_posteriors: bool = False,
            return_features: bool = False
        ) -> (
        Tuple[torch.Tensor, torch.Tensor] 
        | Tuple[torch.Tensor, torch.Tensor, DiagonalGaussianDistribution, DiagonalGaussianDistribution]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ):
        '''
        Decode VAE features obtained through rasterization into color and depth-maps, using the VAE decoder.
        '''
        if return_high_res_posteriors:
            high_res_color_posterior = DiagonalGaussianDistribution(mean=output.color_posterior.mean, logvar=output.color_posterior.logvar)
            high_res_depth_posterior = DiagonalGaussianDistribution(mean=output.depth_posterior.mean, logvar=output.depth_posterior.logvar)

        # First, downsample the latent fields / latent posteriors to match the latent resolution.
        output.color_posterior = self.first_stage.downsample_latent_dist(output.color_posterior, router='appearance')
        output.depth_posterior = self.first_stage.downsample_latent_dist(output.depth_posterior, router='geometry')

        # Sample from the color and depth posteriors.
        color_feature = output.color_posterior.sample() if self.variational_features else output.color_posterior.mode()
        if sample_depth_mode:
            depth_feature = output.depth_posterior.mode()
        else:
            depth_feature = output.depth_posterior.sample() if self.variational_features else output.depth_posterior.mode()

        color_recon = self.first_stage.decode(color_feature, router='appearance')
        color_recon = torch.clamp((color_recon + 1.0) / 2.0, min=0.0, max=1.0)
        
        near = rearrange(batch["target"]["near"], "b v -> (b v)")
        far = rearrange(batch["target"]["far"], "b v -> (b v)")

        depth_recon = self.first_stage.decode(depth_feature, router='geometry')
        depth_recon = near[:, None, None, None] + (depth_recon + 1) * 0.5 * (far[:, None, None, None] - near[:, None, None, None])
        depth_recon = torch.maximum(torch.minimum(depth_recon, far[:, None, None, None]), near[:, None, None, None])
        
        if return_high_res_posteriors:
            return color_recon, depth_recon, high_res_color_posterior, high_res_depth_posterior
        if return_features:
            return color_recon, depth_recon, color_feature, depth_feature

        return color_recon, depth_recon

    @staticmethod
    def _align_depth_with_rasterized(
        pred_depth: Tensor, rasterized_depth: Tensor, opacity: Tensor | None = None, opacity_threshold: float = 0.5
    ) -> Tensor:
        """Align raw diffusion depth to the Gaussian-splat rasterized depth via a per-scene log-affine fit. 
        opacity below opacity_threshold is excluded from the fit.
        pred_depth, rasterized_depth, opacity: (B, V, H, W). Returns depth_aligned: (B, V, 1, H, W).
        """
        valid_mask = (opacity.unsqueeze(2) > opacity_threshold) if opacity is not None else None
        scale, offset = fit_log_affine_scene(
            pred_depth.unsqueeze(2), rasterized_depth.unsqueeze(2), valid_mask=valid_mask
        )
        return apply_log_affine_scene(pred_depth.unsqueeze(2), scale, offset)

    def training_step(self, batch, batch_idx):
        """One training step: for train_cfg.stage='vae', manually alternates generator/
        discriminator updates; for 'diffusion', computes the diffusion loss with automatic
        optimization."""
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # "diffusion" stage uses automatic optimization (see __init__), so it never touches
        # optimizers directly here -- Lightning handles backward/step/clip/scheduler on its own.
        if self.train_cfg.stage == 'vae':
            # Get the optimizers.
            opt = self.optimizers()
            if isinstance(opt, list):
                if len(opt) == 2:
                    # We only train with two
                    assert self.train_cfg.stage == 'vae'
                    g_opt, d_opt = opt
                    # Do not increment global step for discriminator step
                    d_opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
                    d_opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
                else:
                    g_opt, d_opt = opt[0], None
            else:
                g_opt, d_opt = opt, None

            # Run the model.
            # First, run generator pass.
            self.toggle_optimizer(g_opt)

        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        if self.train_cfg.stage == 'vae':
            target_gt = batch["target"]["image"]
            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Generator step for VAE + aux. loss

        # Compute and log loss.
        # Log everything whather the training stage, so that we can easily see when losses are disabled.
        total_loss = 0
        vae_loss = 0
        aux_loss = 0
        diffusion_loss = 0
        
        if self.train_cfg.stage == 'vae':
            # Get decoded color and depth from rasterized latents.
            color_decoded, depth_decoded = self.decode_rasterized_latent(batch, output, h_latent=h//8, w_latent=w//8)
            
            output.color_decoded = color_decoded
            output.depth_decoded = depth_decoded

            # Compute aux + vae losses.
            for loss_name, loss_fn in self.losses["aux_losses"].items():
                loss = loss_fn(output, batch, gaussians, self.global_step)
                self.log(f"loss/{loss_name}", loss)
                aux_loss = aux_loss + loss
            self.log("loss/aux", aux_loss)

            color_rec_loss = 0
            for loss_name, loss_fn in self.losses["vae_losses"]["color_recon"].items():
                loss = loss_fn(output, batch, gaussians, self.first_stage, self.global_step)
                self.log(f"loss/{loss_name}", loss)
                color_rec_loss = color_rec_loss + loss
            self.log("loss/color_recon", color_rec_loss)
            vae_loss += color_rec_loss

            depth_rec_loss = 0
            for loss_name, loss_fn in self.losses["vae_losses"]["depth_recon"].items():
                loss = loss_fn(output, batch, gaussians, self.first_stage, self.global_step)
                self.log(f"loss/{loss_name}", loss)
                depth_rec_loss = depth_rec_loss + loss
            self.log("loss/depth_recon", depth_rec_loss)
            vae_loss += depth_rec_loss

            kl_color_loss = 0
            kl_depth_loss = 0
            if "kl_feature" in self.losses["vae_losses"]:
                kl_loss_fn = self.losses["vae_losses"]["kl_feature"]
                # kl_loss_fn here returns two terms, the first for the color latents, the second for the depth latents
                kl_color_loss, kl_depth_loss = kl_loss_fn(output, batch, gaussians, self.first_stage, self.global_step)

            self.log("loss/kl_color_loss", kl_color_loss)
            self.log("loss/kl_depth_loss", kl_depth_loss)

            vae_loss += kl_color_loss + kl_depth_loss

            generator_loss = 0
            if "gan" in self.losses["vae_losses"]:
                gan_loss_fn = self.losses["vae_losses"]["gan"]
                if gan_loss_fn is not None:
                    generator_loss = gan_loss_fn(output, batch, gaussians, self.first_stage, color_rec_loss, self.global_step)
            self.log("loss/generator", generator_loss)

            total_loss = aux_loss + vae_loss + generator_loss

        elif self.train_cfg.stage == 'diffusion':
            # First, downsample the latent fields / latent posteriors to match the latent resolution.
            output.color_posterior = self.first_stage.downsample_latent_dist(output.color_posterior, router='appearance')
            output.depth_posterior = self.first_stage.downsample_latent_dist(output.depth_posterior, router='geometry')

            if self.diffuser is not None:
                out: DiffuserOutput = self.diffuser(batch, output)

                # Expected only one loss at this stage.
                for loss_name, loss_fn in self.losses["diff_losses"].items():
                    loss = loss_fn(out, batch, self.global_step)
                    self.log(f"loss/{loss_name}", loss)
                    diffusion_loss = diffusion_loss + loss
                self.log("loss/diffusion", diffusion_loss)
                total_loss = diffusion_loss
        else:
            raise ValueError(f'Invalid training stage: {self.train_cfg.stage}')

        if self.train_cfg.stage == 'vae':
            if isinstance(total_loss, Tensor):
                # Need to clip gradients manually as manual optimization is on.
                if not total_loss.isnan().any():
                    g_opt.zero_grad()
                    self.manual_backward(total_loss)
                    self.clip_gradients(
                        g_opt,
                        gradient_clip_val=self.optimizer_cfg.generator.gradient_clip_val,
                        gradient_clip_algorithm="norm"
                    )
                    g_opt.step()
                else:
                    warn(f"Encountered nan generator loss in iteration {self.step_tracker.get_step()}")

            self.untoggle_optimizer(g_opt)
            # Second optimization step: run discriminator
            if (
                d_opt is not None and
                self.first_stage.gan_discriminator is not None and
                'gan' in self.losses['vae_losses'] and
                self.losses['vae_losses']['gan'] is not None and
                self.losses['vae_losses']['gan'].is_adversarial_active(self.global_step)
                ):
                # Discriminator step
                self.toggle_optimizer(d_opt)
                discriminator_loss = self.losses['vae_losses']['gan'].discriminator_step_loss(
                    output, batch, self.first_stage, self.global_step
                )
                self.log("loss/discriminator", discriminator_loss)
                if isinstance(discriminator_loss, Tensor):
                    if not discriminator_loss.isnan().any():
                        d_opt.zero_grad()
                        self.manual_backward(discriminator_loss)
                        # Clip gradients manually
                        self.clip_gradients(
                            d_opt,
                            gradient_clip_val=self.optimizer_cfg.discriminator.gradient_clip_val,
                            gradient_clip_algorithm="norm"
                        )
                        d_opt.step()
                    else:
                        warn(f"Encountered nan discriminator loss in iteration {self.step_tracker.get_step()}")
                self.untoggle_optimizer(d_opt)
            else:
                discriminator_loss = 0
        else:
            discriminator_loss = 0

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):  
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"aux loss = {aux_loss}; "
                f"vae loss = {vae_loss}; "
                f"discriminator loss = {discriminator_loss}; "
                f"diffusion loss = {diffusion_loss}; "
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.train_cfg.stage == 'vae':
            # Do all scheduler steps (manually because of manual optimization).
            schedulers = self.lr_schedulers()
            if schedulers:
                if isinstance(schedulers, list):
                    for scheduler in schedulers:
                        scheduler.step()
                else:
                    schedulers.step()
            return None

        if isinstance(total_loss, Tensor) and not total_loss.isnan().any():
            return total_loss
        if isinstance(total_loss, Tensor):
            warn(f"Encountered nan diffusion loss in iteration {self.step_tracker.get_step()}")
        return None

    def test_step(self, batch, batch_idx):
        """Run one eval batch and write its outputs (color/depth/cameras, plus results.json if
        test_cfg.compute_scores) under test_cfg.output_path/<scene>/. Handles both the standard and
        chunked (test_cfg.sample_in_chunks) sampling strategies."""
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        # name = get_cfg()["wandb"]["name"]
        # path = self.test_cfg.output_path / name
        path = self.test_cfg.output_path
        device = batch["target"]["image"].device

        # Support for resume: if every scene in this batch already has a results.json (written by a
        # previous, possibly crashed/killed run), skip re-computing it entirely.
        if self.test_cfg.compute_scores and all(
            (path / scene / "results.json").exists() for scene in batch["scene"]
        ):
            return

        for idx, scene in enumerate(batch["scene"]):
            context_ids = batch["context"]["index"][idx].cpu().tolist()
            for index, context_image in zip(context_ids, batch["context"]["image"][idx]):
                save_image(context_image, path / scene / f"context_image_{index:0>6}.png")

        # Hook the raw depth predicted by cost-volume depth predictor for the context views (before they are
        # unprojected into 3D Gaussians), needed by pcd_alignment_source="rasterized_context_views".
        capture_context_depth = (
            self.test_cfg.generate_pcd_projections
            and self.test_cfg.pcd_alignment_source == "rasterized_context_views"
        )
        visualization_dump = {} if capture_context_depth else None

        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
            )

        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )
        
        images_prob = output.color
        rgb_gt = batch["target"]["image"]
        depth_gt = batch["target"].get("depth")

        # Decode the predicted rasterized latents.
        color_recons, depth_recons, _, _ = self.decode_rasterized_latent(
            batch, 
            output, 
            h_latent=h//8, 
            w_latent=w//8,
            return_high_res_posteriors=False,
            return_features=True
        )

        # Get the batch dim back.
        color_recons = rearrange(color_recons, "(b v) c h w -> b v c h w", b=b, v=v)
        depth_recons = rearrange(depth_recons, "(b v) c h w -> b v c h w", b=b, v=v)

        # Take only the first channel to visualize depth.
        depth_recons = depth_recons[:, 0, ...]

        color_denoised = None
        depth_denoised = None
        if self.diffuser is not None:
            downsample_factor = 4 if self.diffuser.cfg.upsample_latents else 8
            latent_h, latent_w = h // downsample_factor, w // downsample_factor

            # Use the same noise for all the target views.
            if self.test_cfg.fix_noise_for_targets:
                color_noise = torch.randn((b, 4, latent_h, latent_w)).to(device)
                depth_noise = torch.randn((b, 4, latent_h, latent_w)).to(device)
        
                color_noise = color_noise.unsqueeze(1).expand(b, v, 4, latent_h, latent_w).clone()
                depth_noise = depth_noise.unsqueeze(1).expand(b, v, 4, latent_h, latent_w).clone()
            else:
                color_noise = None
                depth_noise = None

            if self.test_cfg.sample_in_chunks:
                assert b == 1, "Chunked sampling only supports batch size 1."
                scene = batch["scene"][0]
                windows = make_windows(batch["target"]["image"].shape[1], self.test_cfg.chunk_size, self.test_cfg.chunk_overlap, allow_short_last=True)

                cam_dict = {
                    "intrinsics": batch["target"]["intrinsics"][0].cpu(),
                    "extrinsics": batch["target"]["extrinsics"][0].cpu(),
                    "target_index": batch["target"]["index"][0].cpu(),
                    "context_index": batch["context"]["index"][0].cpu(),
                    "near": batch["target"]["near"][0].cpu(),
                    "far": batch["target"]["far"][0].cpu(),
                }
                os.makedirs(path / scene / "cam_dict", exist_ok=True)
                torch.save(cam_dict, path / scene / "cam_dict/cameras.torch")

                if color_noise is None:
                    # Sample Gaussian Noise for all the target frames once, so that we can re-use the same noise for overlapping frames.
                    color_noise = torch.randn((b, v, 4, latent_h, latent_w)).to(device)
                    depth_noise = torch.randn((b, v, 4, latent_h, latent_w)).to(device)

                color_chunks = []
                depth_chunks = []

                for window_idx, window in enumerate(windows):
                    # Build a step batch with the content of a chunk.
                    step_batch = {
                        "context": batch["context"],
                        "target": {
                            "image": batch["target"]["image"][:, window],
                            **({"depth": batch["target"]["depth"][:, window]}
                               if depth_gt is not None else {}),
                            "extrinsics": batch["target"]["extrinsics"][:, window],
                            "intrinsics": batch["target"]["intrinsics"][:, window],
                            "near": batch["target"]["near"][:, window],
                            "far": batch["target"]["far"][:, window],
                            "index": batch["target"]["index"][:, window]
                        },
                        "scene": batch["scene"]
                    }
                    # Build a partial DecoderOutput struct.
                    step_output = DecoderOutput(
                        color = output.color[:, window],
                        depth = output.depth[:, window],
                        mask = output.mask[:, window],
                        color_posterior = DiagonalGaussianDistribution(
                            mean=output.color_posterior.mean[window],
                            logvar=output.color_posterior.logvar[window]
                        ),
                        depth_posterior = DiagonalGaussianDistribution(
                            mean=output.depth_posterior.mean[window],
                            logvar=output.depth_posterior.logvar[window]
                        )
                    )

                    sample_output = self.diffuser.sample(
                        step_batch,
                        step_output,
                        return_dict=True,
                        color_noise=color_noise[:, window],
                        depth_noise=depth_noise[:, window],
                        skip_decode=True,
                    )
                    color_latents_raw = sample_output["color_latents_raw"][0]
                    depth_latents_raw = sample_output["depth_latents_raw"]
                    depth_latents_raw = depth_latents_raw[0] if depth_latents_raw is not None else None

                    if self.test_cfg.chunk_decode_mode == "save_latents":
                        window_chunk = {
                            'scene': scene,
                            'frame_ids': list(window),
                            'color_latents': color_latents_raw.cpu(),
                            'depth_latents': depth_latents_raw.cpu() if depth_latents_raw is not None else None,
                            'near': step_batch["target"]["near"][0].cpu(),
                            'far': step_batch["target"]["far"][0].cpu(),
                        }
                        os.makedirs(path / scene / "latent_windows", exist_ok=True)
                        torch.save(window_chunk, path / scene / f"latent_windows/window_{window_idx:04d}.torch")
                    else:
                        color_chunks.append((list(window), color_latents_raw))
                        if depth_latents_raw is not None:
                            depth_chunks.append((list(window), depth_latents_raw))

                    # Save GT color.
                    for index, color in zip(step_batch["target"]["index"][0], step_batch["target"]["image"][0]):
                        save_image(color, path / scene / f"color_gt/{index:0>6}.png")

                    # Save "GT" depth, if the dataset provides it.
                    if depth_gt is not None:
                        for index, depth in zip(step_batch["target"]["index"][0], step_batch["target"]["depth"][0]):
                            save_image(render_depth_map(depth), path / scene / f"depth_gt/{index:0>6}.png")
                            torch.save(depth, path / scene / f"depth_gt/{index:0>6}.pt")

                # NOTE: Not fully tested or used for our experiments. This is a fallback if you want to generate longer camera trajectories on smaller GPUs.
                if self.test_cfg.chunk_decode_mode == "blend_then_decode":
                    # Blend overlapping chunks at the latent level, then decode the whole stitched scene once.
                    positions, stitched_color_latents = stitch_chunks(color_chunks)
                    target_indices = batch["target"]["index"][0, positions]

                    color_denoised = self.diffuser.last_stage_decode(
                        stitched_color_latents.unsqueeze(0),
                        router="vanilla",
                        normalization="image",
                        image_set=False,
                    )[0]
                    for index, color in zip(target_indices, color_denoised):
                        save_image(color, path / scene / f"color/{index:0>6}.png")

                    if depth_chunks:
                        _, stitched_depth_latents = stitch_chunks(depth_chunks)
                        depth_denoised = self.diffuser.last_stage_decode(
                            stitched_depth_latents.unsqueeze(0),
                            router="vanilla",
                            normalization=None if self.test_cfg.generate_pcd_projections else "depth",
                            near=batch["target"]["near"][:, positions],
                            far=batch["target"]["far"][:, positions],
                            image_set=False,
                        )[0]
                        for index, depth in zip(target_indices, depth_denoised):
                            save_image(render_depth_map(depth), path / scene / f"depth/{index:0>6}.png")
                            torch.save(depth, path / scene / f"depth/{index:0>6}.pt")

            else:
                sample_output = self.diffuser.sample(
                    batch,
                    output,
                    return_dict=True,
                    color_noise=color_noise,
                    depth_noise=depth_noise,
                    depth_normalization=False if self.test_cfg.generate_pcd_projections else True
                )
                color_denoised = sample_output['color_denoised']
                depth_denoised = sample_output['depth_denoised']

                if self.test_cfg.generate_pcd_projections:
                    assert b == 1
                    assert depth_denoised is not None, \
                        "generate_pcd_projections requires depth modeling to be enabled (disable_depth_modeling=False)."

                    # Render projections and construct projection image.
                    alignment_source = self.test_cfg.pcd_alignment_source

                    if alignment_source == "rasterized_all_views":
                        depth_aligned = self._align_depth_with_rasterized(depth_denoised, output.depth, opacity=output.mask)
                    else:
                        context_ids = batch["context"]["index"][0].cpu().tolist()
                        target_ids = batch["target"]["index"][0].cpu().tolist()
                        missing = [x for x in context_ids if x not in target_ids]
                        assert not missing, \
                            f"pcd_alignment_source={alignment_source!r} requires context views to be a " \
                            f"subset of target views; missing context indices {missing} from target " \
                            f"indices {target_ids}."
                        indices = [target_ids.index(x) for x in context_ids]

                        context_pred = depth_denoised[0][indices].unsqueeze(0)

                        if alignment_source == "oracle":
                            # Compute affine transform params based on our predictions for context
                            # views vs. "oracle" predictions (VGGT).
                            context_ref = batch["context"].get("depth")
                            assert context_ref is not None, \
                                "pcd_alignment_source='oracle' requires the dataset to be configured " \
                                "with load_depth_labels=true."
                        elif alignment_source == "rasterized_context_views":
                            # Oracle-free strategy: compare the diffusion prediction at context views
                            # against the depth predicted by the cost-volume depth predictor for the
                            # context views themselves (before it is unprojected into 3D Gaussians).
                            context_ref = visualization_dump["depth"].mean(dim=(-2, -1))
                        else:
                            raise ValueError(f"Invalid pcd_alignment_source: {alignment_source}")

                        scale, offset = fit_log_affine_scene(context_pred.unsqueeze(2), context_ref.unsqueeze(2))
                        depth_aligned = apply_log_affine_scene(depth_denoised.unsqueeze(2), scale, offset)

                    K_pix = batch["target"]["intrinsics"].clone()
                    K_pix[..., 0, 0] *= w
                    K_pix[..., 1, 1] *= h
                    K_pix[..., 0, 2] *= w
                    K_pix[..., 1, 2] *= h

                    projections = render_pointcloud_projections_from_rgbd(
                        color_denoised,
                        depth_aligned,
                        batch["target"]["extrinsics"],
                        K_pix,
                        resolution=256
                    )

                    panels = [add_label(hcat(*projections[0]), "Pred. PCD")]
                    if depth_gt is not None:
                        gt_projections = render_pointcloud_projections_from_rgbd(
                            rgb_gt,
                            depth_gt.unsqueeze(2),
                            batch["target"]["extrinsics"],
                            K_pix,
                            resolution=256
                        )
                        panels.append(add_label(hcat(*gt_projections[0]), "GT PCD"))
                    projections = vcat(*panels)
                    scene = batch["scene"][0]
                    save_image(projections, path / scene / f"pcd_proj.png")
                else:
                    depth_aligned = None

                # Save images for computing FID scores later.
                if self.test_cfg.save_image:

                    # Using same near and far range for visualization.
                    near = batch["target"]["near"][0][0].item()
                    far = batch["target"]["far"][0][0].item()

                    for idx, scene in enumerate(batch["scene"]):
                        for index, color_gt in zip(batch["target"]["index"][idx], rgb_gt[idx]):
                            if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][idx]):
                                continue
                            save_image(color_gt, path / scene / f"color_gt/{index:0>6}.png")

                        for index, color_gsplat in zip(batch["target"]["index"][idx], images_prob[idx]):
                            if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][idx]):
                                continue
                            save_image(color_gsplat, path / scene / f"color/{index:0>6}.png")

                        if self.test_cfg.save_intermediate_recons:
                            for index, color_recon in zip(batch["target"]["index"][idx], color_recons[idx]):
                                if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][idx]):
                                    continue
                                save_image(color_recon, path / scene / f"color_recon/{index:0>6}.png")

                            for index, depth_recon in zip(batch["target"]["index"][idx], depth_recons[idx]):
                                if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][idx]):
                                    continue
                                save_image(render_depth_map(depth_recon), path / scene / f"depth_recon/{index:0>6}.png")

                        if color_denoised is not None:
                            for index, color in zip(batch["target"]["index"][idx], color_denoised[idx]):
                                if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][idx]):
                                    continue
                                save_image(color, path / scene / f"color_denoised/{index:0>6}.png")

                        if depth_denoised is not None:
                            for index, depth in zip(batch["target"]["index"][idx], depth_denoised[idx]):
                                if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][idx]):
                                    continue
                                save_image(render_depth_map(depth), path / scene / f"depth_denoised/{index:0>6}.png")
                                torch.save(depth, path / scene / f"depth_denoised/raw_{index:0>6}.torch")
                        
                        if depth_aligned is not None:
                            depth_aligned = depth_aligned.squeeze(2)  # remove the extra dim
                            for index, depth in zip(batch["target"]["index"][idx], depth_aligned[idx]):
                                if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][idx]):
                                    continue
                                save_image(render_depth_map(depth), path / scene / f"depth_aligned/{index:0>6}.png")
                                torch.save(depth, path / scene / f"depth_aligned/raw_{index:0>6}.torch")
                        
                        # Save camera params for later visualization in viser.
                        cam_dict = {
                            "intrinsics": batch["target"]["intrinsics"][idx].cpu(),
                            "extrinsics": batch["target"]["extrinsics"][idx].cpu(),
                        }
                        os.makedirs(path / scene / "cam_dict", exist_ok=True)
                        torch.save(cam_dict, path / scene / "cam_dict/cameras.torch")

                # compute scores
                if self.test_cfg.compute_scores:
                    if batch_idx < self.test_cfg.eval_time_skip_steps:
                        self.time_skip_steps_dict["encoder"] += 1
                        self.time_skip_steps_dict["decoder"] += v

                    for idx, scene in enumerate(batch['scene']):
                        # Compute per-image metrics, average per scene, and save both to disk.
                        target_ids = batch['target']['index'][idx].cpu().tolist()
                        psnr = compute_psnr(rgb_gt[idx], color_denoised[idx])
                        ssim = compute_ssim(rgb_gt[idx], color_denoised[idx])
                        lpips = compute_lpips(rgb_gt[idx], color_denoised[idx])
                        dists = compute_dists(rgb_gt[idx], color_denoised[idx])

                        per_image = {
                            str(target_id): {
                                "psnr": psnr[i].item(),
                                "ssim": ssim[i].item(),
                                "lpips": lpips[i].item(),
                                "dists": dists[i].item(),
                            }
                            for i, target_id in enumerate(target_ids)
                        }
                        full_dict = {
                            "scene": scene,
                            "context_ids": batch['context']["index"][idx].cpu().tolist(),
                            "target_ids": target_ids,
                            "psnr": psnr.mean().item(),
                            "ssim": ssim.mean().item(),
                            "lpips": lpips.mean().item(),
                            "dists": dists.mean().item(),
                            "per_image": per_image,
                        }
                        with (path / scene / "results.json").open("w") as f:
                            json.dump(full_dict, f, indent=2)

    def on_test_end(self) -> None:
        """After all test batches: aggregate per-scene results.json into overall scores (unless
        chunked), and optionally fit the point-cloud scale."""
        out_dir = self.test_cfg.output_path

        if not self.test_cfg.sample_in_chunks:
            saved_scores = {"denoised": {}}

            if self.test_cfg.compute_scores:
                self.benchmarker.dump_memory(out_dir / "peak_memory.json")
                self.benchmarker.dump(out_dir / "benchmark.json")

                # Aggregate metrics pre-computed per scene in */results.json.
                all_results = []
                metric_scores = {"psnr": [], "ssim": [], "lpips": [], "dists": []}
                for results_path in sorted(out_dir.glob("*/results.json")):
                    with results_path.open("r") as f:
                        full_dict = json.load(f)
                    all_results.append(full_dict)
                    for metric_name in metric_scores:
                        metric_scores[metric_name].append(full_dict[metric_name])

                for metric_name, scores in metric_scores.items():
                    avg_scores = sum(scores) / len(scores)
                    saved_scores["denoised"][metric_name] = avg_scores
                    print(f"denoised_{metric_name}", avg_scores)
                    with (out_dir / f"scores_denoised_{metric_name}_all.json").open("w") as f:
                        json.dump(scores, f)

                for tag, times in self.benchmarker.execution_times.items():
                    times = times[int(self.time_skip_steps_dict[tag]) :]
                    saved_scores[tag] = [len(times), np.mean(times)]
                    print(
                        f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                    )
                    self.time_skip_steps_dict[tag] = 0

                with (out_dir / f"all_results.json").open("w") as f:
                    json.dump(all_results, f, indent=2)

                with (out_dir / f"scores_all_avg.json").open("w") as f:
                    json.dump(saved_scores, f, indent=2)

                self.benchmarker.clear_history()
            else:
                self.benchmarker.dump(self.test_cfg.output_path / "benchmark.json")
                self.benchmarker.dump_memory(
                    self.test_cfg.output_path / "peak_memory.json"
                )
                self.benchmarker.summarize()

        if self.test_cfg.fit_pcd_scale and self.trainer.is_global_zero:   # only rank 0, for DDP
            from ..scripts.fit_context_scale import fit_all_scenes
            print(f"[on_test_end] fitting point-cloud scale over {out_dir} ...")
            fit_all_scenes(out_dir, device=str(self.device))

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        """Render a validation batch and log color/depth comparison images (and, depending on
        config, wobble/interpolation videos) to wandb."""
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]
        
        near = rearrange(batch["target"]["near"], "b v -> (b v)")
        far = rearrange(batch["target"]["far"], "b v -> (b v)")

        color_denoised = None
        depth_denoised = None
        if self.train_cfg.stage == 'vae':
            color_recons, depth_recons = self.decode_rasterized_latent(batch, output_softmax, h_latent=h//8, w_latent=w//8)
            # Take only the first channel to visualize depth.
            depth_recons = depth_recons[:, 0, ...]
        else:
            assert self.diffuser is not None
            # First, downsample the latent fields / latent posteriors to match the latent resolution.
            output_softmax.color_posterior = self.first_stage.downsample_latent_dist(output_softmax.color_posterior, router='appearance')
            output_softmax.depth_posterior = self.first_stage.downsample_latent_dist(output_softmax.depth_posterior, router='geometry')

            sample_output = self.diffuser.sample(
                batch,
                output_softmax,
                return_dict=True,
                color_noise=None,
                depth_noise=None,
            )

            color_denoised = sample_output['color_denoised'][0]
            depth_denoised = sample_output['depth_denoised']
            if depth_denoised is not None:
                depth_denoised = depth_denoised[0]
            
            latent_color_sample = sample_output['latent_color_sample']
            latent_depth_sample = sample_output['latent_depth_sample']

            color_recons = self.first_stage.decode(latent_color_sample, router='appearance')
            color_recons = torch.clamp((color_recons + 1.0) / 2.0, min=0.0, max=1.0)

            depth_recons = self.first_stage.decode(latent_depth_sample, router='geometry')
            depth_recons = near[:, None, None, None] + (depth_recons + 1) * 0.5 * (far[:, None, None, None] - near[:, None, None, None])
            depth_recons = torch.maximum(torch.minimum(depth_recons, far[:, None, None, None]), near[:, None, None, None])
            depth_recons = depth_recons[:, 0, ...]

        # Compute validation metrics for color/appearance.
        tags = ("val_rast_rgb", "val_rast_feat_color",  "val_denoised_color") if self.train_cfg.stage == 'diffusion' else ("val_rast_rgb", "val_rast_feat_color",)
        rgbs = (rgb_softmax, color_recons, color_denoised, ) if self.train_cfg.stage == 'diffusion' else (rgb_softmax, color_recons,)

        rgb_gt = batch["target"]["image"][0]
        depth_gt = batch["target"]["depth"][0]
        depth_gt = torch.maximum(torch.minimum(depth_gt, far[:, None, None]), near[:, None, None])

        for tag, rgb in zip(tags, rgbs):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Using same near and far range for visualization.
        near = batch["target"]["near"][0][0].item()
        far = batch["target"]["far"][0][0].item()

        images = [
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target Color (Ground Truth)"),
            add_label(vcat(*[render_depth_map(depth_gt[i]) for i in range(depth_gt.shape[0])]), "Target Depth (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
            add_label(vcat(*color_recons), "Color Recon. (GSplat Features)"),
            add_label(vcat(*[render_depth_map(depth_recons[i]) for i in range(depth_recons.shape[0])]), "Depth Recon. (GSplat Features)")
        ]

        if self.train_cfg.stage == 'diffusion':
            images.append(
                add_label(vcat(*color_denoised), "Color Denoised")
            )
            if depth_denoised is not None:
                images.append(
                    add_label(vcat(*[render_depth_map(depth_denoised[i]) for i in range(depth_denoised.shape[0])]), "Depth Denoised")
                )

        # Construct comparison image.
        comparison = hcat(*images)
        image = prep_image(add_border(comparison))
        print(
            "log_image",
            "batch_idx=", batch_idx,
            "global_step=", self.global_step,
            "image_shape=", image.shape,
            "scene=", batch["scene"],
            "scene_len=", len(batch["scene"]) if hasattr(batch["scene"], "__len__") else None,
        )

        self.logger.log_image(
            "comparison",
            [image],
            step=self.global_step,
            caption=batch["scene"],
        )

        if depth_denoised is not None:
            # Render projections and construct projection image.
            K_pix = batch["target"]["intrinsics"].clone()
            K_pix[..., 0, 0] *= w
            K_pix[..., 1, 1] *= h
            K_pix[..., 0, 2] *= w
            K_pix[..., 1, 2] *= h

            if self.test_cfg.align_pcd_with_rasterized_depth:
                depth_for_pcd = self._align_depth_with_rasterized(
                    depth_denoised.unsqueeze(0), output_softmax.depth[0].unsqueeze(0),
                    opacity=output_softmax.mask[0].unsqueeze(0),
                )
            else:
                depth_for_pcd = depth_denoised.unsqueeze(1).unsqueeze(0)

            projections = render_pointcloud_projections_from_rgbd(
                color_denoised.unsqueeze(0),
                depth_for_pcd,
                batch["target"]["extrinsics"],
                K_pix,
                resolution=256
            )

            gt_projections = render_pointcloud_projections_from_rgbd(
                rgb_gt.unsqueeze(0),
                depth_gt.unsqueeze(1).unsqueeze(0),
                batch["target"]["extrinsics"],
                K_pix,
                resolution=256
            )
            # projections = hcat(*projections[0])
            # gt_projections = hcat(*gt_projections[0])
            projections = [
                add_label(hcat(*projections[0]), "Pred. PCD"),
                add_label(hcat(*gt_projections[0]), "GT PCD"),
            ]
            projections = vcat(*projections)

            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        # but only log if wandb is enabled.
        if self.test_cfg.generate_val_video_wobble and wandb.run is not None:
            self.render_video_wobble(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        """Render a small circular "wobble" around the first context view (needs exactly 2
        context views, to get a wobble radius from their baseline)."""
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(
            batch, trajectory_fn, "wobble", num_frames=self.test_cfg.val_video_num_frames
        )

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        """Render a smooth interpolation between the first two context views (or the first
        context and first target view, if there is only one context view)."""
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        """Like render_video_interpolation, but with an exaggerated wobble added on top (needs
        exactly 2 context views)."""
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        downscale_factor: float = 4.0,  # downscale video resolution
    ) -> None:
        """Render a video along trajectory_fn's camera path and log it to wandb as `name`."""
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob in images_prob
        ]

        # Render denoised video.

        # Downsample the latent fields / latent posteriors to match the latent resolution.
        output_prob.color_posterior = self.first_stage.downsample_latent_dist(output_prob.color_posterior, router='appearance')
        output_prob.depth_posterior = self.first_stage.downsample_latent_dist(output_prob.depth_posterior, router='geometry')

        batch["target"]["extrinsics"] = extrinsics
        batch["target"]["intrinsics"] = intrinsics
        batch["target"]["near"] = near
        batch["target"]["far"] = far

        diffusion_output = self.diffuser.sample(
            batch,
            output_prob,
            return_dict=True,
            decode_video=True
        )
        color_denoised = diffusion_output["color_denoised"]
        depth_denoised = diffusion_output["depth_denoised"]
        
        if depth_denoised is not None:
            images_denoised = [
                vcat(rgb, depth)
                for rgb, depth in zip(color_denoised[0], depth_map(depth_denoised[0]))
            ]
        else:
            images_denoised = [rgb for rgb in color_denoised[0]]

        images_d = [
            add_border(
                hcat(
                    add_label(image_denoised, "Denoised"),
                )
            )
            for image_denoised in images_denoised
        ]

        video = torch.stack(images) 
        denoised_video = torch.stack(images_d)

        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        denoised_video = (denoised_video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()

        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
            denoised_video = pack([denoised_video, denoised_video[::-1][1:-1]], "* c h w")[0]

        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4"),
            f"video/{name}_denoised": wandb.Video(denoised_video[None], fps=30, format="mp4")
        }

        # Since PyTorch Lightning seems to not support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    @staticmethod
    def get_lr_scheduler(
        opt: optim.Optimizer,
        lr_scheduler_cfg: LRSchedulerCfg,
        lr: float,
        max_steps: int,
    ) -> optim.lr_scheduler.LRScheduler:
        """Cosine (OneCycleLR) or linear warmup schedule, per lr_scheduler_cfg.cosine_lr."""
        warm_up_steps = lr_scheduler_cfg.warm_up_steps
        if lr_scheduler_cfg.cosine_lr:
            pct_start = warm_up_steps / max_steps
            return torch.optim.lr_scheduler.OneCycleLR(
                opt, lr,
                max_steps + 10,
                pct_start=pct_start,
                cycle_momentum=False,
                anneal_strategy='cos',
                final_div_factor=lr_scheduler_cfg.final_div_factor,
            )
        return torch.optim.lr_scheduler.LinearLR(
            opt,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

    def configure_optimizers(self):
        """Build the optimizer(s) and LR scheduler(s) for the current training stage ('vae':
        generator [+ discriminator, see the manual-optimization branch below]; 'diffusion':
        a single AdamW optimizer over the diffuser)."""
        optimizers = []
        schedulers = []

        if self.train_cfg.stage == 'vae':
            # Gradient tracking is disabled with _disable_gradient_tracking in model_wrapper.py.
            g_modules = [
                self.encoder,
                self.decoder,  # no actual trainable params
                self.first_stage.app_vae,
                self.first_stage.app_down,
                self.first_stage.geom_vae,
                self.first_stage.geom_down,
            ]
            total_params = sum(p.numel() for m in g_modules for p in m.parameters())
            all_params = [p for m in g_modules for p in m.parameters() if p.requires_grad]

        elif self.train_cfg.stage == 'diffusion':
            total_params = sum(p.numel() for p in self.diffuser.parameters())
            all_params = [p for p in self.diffuser.parameters() if p.requires_grad]
        else:
            raise ValueError(f'Invalid training stage: {self.train_cfg.stage}')

        trainable_params = sum(p.numel() for p in all_params if p.requires_grad)

        print(f"[configure_optimizers] Total parameters: {total_params:,}")
        print(f"[configure_optimizers] Trainable parameters: {trainable_params:,}")

        if self.train_cfg.stage == 'diffusion':
            # Optimizer for the denoiser.
            g_opt = optim.AdamW(all_params, lr=self.generator_lr)
        else:
            # NOTE: use Adam to reproduce exps for multi-view VAE training.
            g_opt = optim.Adam(all_params, lr=self.generator_lr)

        optimizers.append(g_opt)
        # Init VAE generator scheduler.
        # "interval": "step" only matters for automatic optimization (enabled for the "diffusion" stage); manual
        # optimization ("vae") steps its schedulers itself in training_step instead.
        if self.optimizer_cfg.generator.lr_scheduler is not None:
            g_scheduler = self.get_lr_scheduler(
                g_opt, self.optimizer_cfg.generator.lr_scheduler,
                self.optimizer_cfg.generator.lr, self.trainer.max_steps,
            )
            schedulers.append({"scheduler": g_scheduler, "interval": "step", "frequency": 1})

        # Init discriminator optimizer for GAN loss.
        if self.train_cfg.stage == 'vae' and self.optimizer_cfg.discriminator is not None:
            disc_params = [p for p in self.first_stage.gan_discriminator.parameters() if p.requires_grad]
            d_opt = optim.Adam(disc_params, lr=self.discriminator_lr)
            optimizers.append(d_opt)
            if self.optimizer_cfg.discriminator.lr_scheduler is not None:
                d_scheduler = self.get_lr_scheduler(
                    d_opt, self.optimizer_cfg.discriminator.lr_scheduler,
                    self.optimizer_cfg.discriminator.lr, self.trainer.max_steps,
                )
                schedulers.append({"scheduler": d_scheduler, "interval": "step", "frequency": 1})

        return optimizers, schedulers

    # Define some useful hooks for diffusion inference.
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.diffuser is not None and self.diffuser.active(self.global_step):
            if self.global_rank == 0:
                print("Setting max timesteps for diffusion validation to: ", self.diffuser.cfg.train_scheduler.num_inference_steps)
            self.diffuser.set_inference_timesteps()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.diffuser is not None and self.diffuser.active(self.global_step):
            if self.global_rank == 0:
                print("Setting max timesteps for diffusion testing to: ", self.diffuser.cfg.train_scheduler.num_inference_steps)
            self.diffuser.set_inference_timesteps()

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.diffuser is not None and self.diffuser.active(self.global_step):
            if self.global_rank == 0:
                print("Setting max timesteps for diffusion prediction to: ", self.diffuser.cfg.train_scheduler.num_inference_steps)
            self.diffuser.set_inference_timesteps()

