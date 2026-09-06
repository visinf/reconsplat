import copy
import json
import os
import warnings
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler

os.environ['CUDA_LATENT_RASTERIZER'] = '1'  # Use the CUDA rasterizer from latentSplat as default option.
print('[INFO] Using CUDA latent rasterizer:', os.environ['CUDA_LATENT_RASTERIZER'])

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import LossDict, get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.diffusion_adapter import DiffusionAdapter
    from src.model.encoder import get_encoder
    from src.model.first_stage.adapter import FirstStageAdapter
    from src.model.model_wrapper import ModelWrapper

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


class EMACallback(Callback):
    """
    Exponential Moving Average Callback for diffusion models.

    Keeps an EMA copy of the model weights:
        - updated after optimizer step
        - used for validation / sampling
        - saved/restored in checkpoints

    Works correctly under DDP.
    """

    def __init__(
        self,
        decay: float = 0.999,
        update_interval: int = 1,
        warmup_steps: int = 0,
        use_ema_for_val: bool = True,
    ):
        super().__init__()

        self.decay = decay
        self.update_interval = update_interval
        self.warmup_steps = warmup_steps
        self.use_ema_for_val = use_ema_for_val
        self.ema_model = None
        self._last_ema_step = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA model copy at training start."""

        if self.ema_model is None:
            self.ema_model = copy.deepcopy(pl_module.diffuser.denoiser)
            # EMA model should never require gradients
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
            # Move EMA model to correct device
            self.ema_model.to(pl_module.device)

        if trainer.is_global_zero:
            print(f"[EMA] Initialized EMA model copy (decay={self.decay})")

    @torch.no_grad()
    def update_ema(self, pl_module):
        """EMA weight update."""
        model = pl_module.diffuser.denoiser
        for p, p_ema in zip(model.parameters(), self.ema_model.parameters(), strict=True):
            if p.requires_grad:
                p_ema.data.mul_(self.decay)
                p_ema.data.add_((1.0 - self.decay) * p.data)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA after each optimizer step."""
        step = trainer.global_step
        if step < self.warmup_steps:
            return
        if step % self.update_interval != 0:
            return
        if step == self._last_ema_step:
            # Non-boundary micro-batch under gradient accumulation: global_step hasn't advanced
            # since the last real optimizer step, so the model weights haven't changed either.
            return
        self._last_ema_step = step
        self.update_ema(pl_module)

    def on_validation_start(self, trainer, pl_module):
        """Swap model → EMA for validation."""
        if not self.use_ema_for_val:
            return
        # Backup training weights
        self.backup_state = copy.deepcopy(pl_module.diffuser.denoiser.state_dict())
        # Load EMA weights
        pl_module.diffuser.denoiser.load_state_dict(self.ema_model.state_dict())

        if trainer.is_global_zero:
            print("[EMA] Swapped EMA weights for validation")

    def on_validation_end(self, trainer, pl_module):
        """Restore original weights after validation."""

        if not self.use_ema_for_val:
            return

        pl_module.diffuser.denoiser.load_state_dict(self.backup_state)

        if trainer.is_global_zero:
            print("[EMA] Restored training weights after validation")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA weights inside checkpoint."""

        checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        if trainer.is_global_zero:
            print("[EMA] Saved EMA weights in checkpoint")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Restore EMA weights from checkpoint."""

        if "ema_state_dict" in checkpoint:
            self.ema_model = copy.deepcopy(pl_module.diffuser.denoiser)
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
            self.ema_model.to(pl_module.device)

            if trainer.is_global_zero:
                print("[EMA] Loaded EMA weights from checkpoint")
        else:
            print("[EMA] Found no EMA state dict in the provided checkpoint.")

class StageCheckpointing(Callback):
    def __init__(self, output_dir, stage, losses: LossDict, every_n_train_steps: int,
                 milestone_every_n_steps: int | None = None,
                 milestone_steps: list[int] | None = None):
        self.output_dir = output_dir
        self.stage = stage
        self.every_n_train_steps = every_n_train_steps
        self.milestone_every_n_steps = milestone_every_n_steps
        self.milestone_steps = set(milestone_steps or ())
        self._saved_milestones = set()

        # Build the checkpointing map based on active losses.
        # This allows to save model checkpoints before a loss becomes active
        # (e.g., before we activate adversarial loss for VAE)
        self.checkpoint_suffix_map = {}

        # Nothing to do if we are in diffusion stage, since we do not have loss scheduling for diffusion.
        if stage == 'vae':
            min_depth_recon = min(losses["vae_losses"]["depth_recon"][loss].cfg.apply_after_step for loss in losses["vae_losses"]["depth_recon"])
            min_color_recon = min(losses["vae_losses"]["color_recon"][loss].cfg.apply_after_step for loss in losses["vae_losses"]["color_recon"])
            vae_starting_step = min(min_color_recon,
                                    min_depth_recon,
                                    losses["vae_losses"]["kl_feature"].cfg.apply_after_step)

            # If the VAE has a starting step > 0, we save a checkpoint before the VAE losses are applied.
            if vae_starting_step > 0:
                # Until this step, we are only training with auxiliary losses.
                self.checkpoint_suffix_map[vae_starting_step] = 'aux_only'

            if 'gan' in losses["vae_losses"]:
                adversarial_starting_step = losses["vae_losses"]["gan"].cfg.apply_after_step
                if adversarial_starting_step > 0:
                    # Until this step, we are only training with VAE reconstruction losses.
                    self.checkpoint_suffix_map[adversarial_starting_step] = 'vae_pre_adv'

    def _save_checkpoint(self, trainer, filepath):
        trainer.save_checkpoint(filepath, weights_only=False)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Check if we need to save a checkpoint before the current step, based on the loss scheduling.
        if trainer.global_step in self.checkpoint_suffix_map:
            self._save_checkpoint(trainer,
                                  filepath=self.output_dir / f'{self.checkpoint_suffix_map[trainer.global_step]}.ckpt')

        # Milestone checkpoints: saved under a distinct filename.
        step = trainer.global_step
        periodic = (
            self.milestone_every_n_steps is not None
            and step > 0
            and step % self.milestone_every_n_steps == 0
        )

        if (periodic or step in self.milestone_steps) and step not in self._saved_milestones:
            self._saved_milestones.add(step)
            self._save_checkpoint(trainer, filepath=self.output_dir / f'milestone_step_{step}.ckpt')

    def on_train_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_train_steps == 0:
            return
        self._save_checkpoint(
            trainer, filepath=self.output_dir / f'epoch_{trainer.current_epoch}-step_{trainer.global_step}.ckpt'
        )

class StripFrozenPretrainedWeights(Callback):
    """
    Drop frozen, off-the-shelf pretrained submodules from saved checkpoints.
    """

    PREFIXES = (
        "first_stage.vanilla_vae.",
        "diffuser.clip_conditioner.",
    )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if key.startswith(self.PREFIXES):
                del state_dict[key]

class PrintParamCounts(Callback):
    def on_fit_start(self, trainer, pl_module):
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        print(f"[after-freeze] Trainable: {trainable:,} / Total: {total:,} "
              f"({100*trainable/total:.3f}%)")

class LogPeakMemory(Callback):
    """
    Print peak CUDA memory per rank at a fixed step interval.
    """

    def __init__(self, every_n_steps: int):
        self.every_n_steps = max(1, every_n_steps)

    def on_train_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available() or trainer.global_step % self.every_n_steps:
            return
        gib = 2 ** 30
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / gib
        print(
            f"[mem] rank {trainer.global_rank} step {trainer.global_step}: "
            f"peak_alloc={torch.cuda.max_memory_allocated() / gib:.1f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved() / gib:.1f}GiB "
            f"/ {total:.0f}GiB",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()

class DumpResultsOnExit(Callback):
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)

    def _dump(self, trainer, pl_module):
        # write only once in DDP
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = getattr(pl_module, "all_results", {})
        with (self.out_dir / "all_results.json").open("w") as f:
            json.dump(data, f, indent=2)

    # called if *any* exception bubbles up through Lightning
    def on_exception(self, trainer, pl_module, err):
        self._dump(trainer, pl_module)

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            f"checkpoints/ldm/{cfg_dict.wandb.name}",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            save_on_exception=True,
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )


    # Set up EMA sampling for the diffusion denoiser.
    if cfg.train.stage == "diffusion":
        callbacks.append(
            EMACallback(warmup_steps=1)
        )
        callbacks.append(
            StripFrozenPretrainedWeights()
        )

    # Set up results dumping in case of any exception, to save intermediate results.
    if cfg.mode == 'test':
        callbacks.append(
            DumpResultsOnExit(out_dir=cfg.test.output_path)
        )

    # Load losses.
    losses = get_losses(cfg.loss)

    # Setup further checkpoinint callback based on loss configuration.
    callbacks.append(
        StageCheckpointing(
            Path(f"checkpoints/ldm/{cfg_dict.wandb.name}"),
            stage=cfg.train.stage,
            losses=losses,
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            milestone_every_n_steps=cfg.checkpointing.milestone_every_n_steps,
            milestone_steps=cfg.checkpointing.milestone_steps,
        )
    )
    callbacks.append(
        PrintParamCounts()
    )
    callbacks.append(
        LogPeakMemory(every_n_steps=cfg.train.print_log_every_n_steps)
    )

    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoints for loading. The first-stage checkpoint holds the encoder/decoder/
    # first_stage weights (produced by the vae stage); the second-stage checkpoint holds the
    # diffuser/denoiser weights (produced by the diffusion stage, unused for the vae stage).
    first_stage_checkpoint_path = update_checkpoint_path(cfg.checkpointing.load_first_stage, cfg.wandb)
    second_stage_checkpoint_path = update_checkpoint_path(cfg.checkpointing.load_second_stage, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    if torch.cuda.device_count() > 1:
        video_finetuning = (
            cfg.train.stage == "diffusion"
            and cfg.model.diffuser is not None
            and getattr(cfg.model.diffuser.denoiser, "video_finetuning", False)
        )
        needs_unused_params = cfg.train.stage == "vae" or video_finetuning
        strategy = "ddp_find_unused_parameters_true" if needs_unused_params else "ddp"
    else:
        strategy = "auto"

    profiler = None
    if cfg.trainer.profiler == "simple":
        profiler = SimpleProfiler(dirpath=output_dir, filename="profiler_report")
    elif cfg.trainer.profiler == "advanced":
        profiler = AdvancedProfiler(dirpath=output_dir, filename="profiler_report")

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=cfg.trainer.devices if cfg.trainer.devices is not None else "auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy=strategy,
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=cfg.mode == "test",
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=1,
        precision=cfg.trainer.precision,
        profiler=profiler,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.optimizer.generator.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(trainer.local_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    first_stage = FirstStageAdapter(
        cfg.model.autoencoder,
        custom_color_decoder_cfg=cfg.model.color_decoder,
        custom_depth_decoder_cfg=cfg.model.depth_decoder,
        discriminator_cfg=cfg.model.discriminator,
        mode=cfg.mode,
        stage=cfg.train.stage,
    )

    # NOTE: The actual processing height and width depends on the patch_shim applied by the encoder.
    image_height, image_width = cfg.dataset.image_shape
    patch_size = cfg.model.encoder.shim_patch_size * cfg.model.encoder.downscale_factor
    image_height = (image_height // patch_size) * patch_size
    image_width = (image_width // patch_size) * patch_size

    if cfg.train.stage == 'diffusion' and cfg.model.diffuser is not None:
        if cfg.model.diffuser.load_ema_weights is None:
            # Unset means 'decide from the mode': evaluation should use the EMA weights (that is
            # what validation used during training), while training must start from the raw weights.
            cfg.model.diffuser.load_ema_weights = cfg.mode == "test"
            print(f'[main] load_ema_weights unset -> {cfg.model.diffuser.load_ema_weights} (mode={cfg.mode})')
        diffuser = DiffusionAdapter(
            cfg.model.diffuser, first_stage, image_height, image_width,
            mode=cfg.mode,
            load_from_ckpt=str(second_stage_checkpoint_path) if second_stage_checkpoint_path is not None else None,
        )
    else:
        diffuser = None

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "first_stage": first_stage,
        "diffuser": None,
        "losses": losses,
        "step_tracker": step_tracker,
    }

    if cfg.train.stage == "vae":
        if cfg.mode == "train" and first_stage_checkpoint_path is not None and not cfg.checkpointing.resume:
            # Just load model weights, without optimizer states
            # e.g., fine-tune from the released weights on other datasets
            model_wrapper = ModelWrapper.load_from_checkpoint(
                first_stage_checkpoint_path, **model_kwargs, strict=True)
            print(cyan(f"Loaded weigths from {first_stage_checkpoint_path}."))
        elif cfg.mode == "test":
            # Discard discriminator when using a pre-trained checkpoint at test time.
            model_wrapper = ModelWrapper(**model_kwargs)
            ckpt = torch.load(first_stage_checkpoint_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            # Filter out first_stage keys.
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not (k.startswith("first_stage.gan_discriminator"))
            }
            missing, unexpected = model_wrapper.load_state_dict(filtered_state_dict, strict=False)
            print("[INFO] Loading base model for diffusion:")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            model_wrapper = ModelWrapper(**model_kwargs)
    else:
        model_wrapper = ModelWrapper(**model_kwargs)
        ckpt = torch.load(first_stage_checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        # Only load first_stage/encoder/decoder keys, so we avoid loading diffuser weights here.
        # Diffuser weights are loaded separately.
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not (k.startswith("diffuser."))
        }
        missing, unexpected = model_wrapper.load_state_dict(filtered_state_dict, strict=False)
        print("[INFO] Loading base model for diffusion:")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    # NOTE: assign diffuser wrapper to model after loading ckpt
    #       also, diffusion ckpt is loaded separately inside the diffusion_adapter init.
    if cfg.train.stage == "diffusion":
        model_wrapper.diffuser = diffuser
        model_wrapper.strict_loading = False

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        if cfg.checkpointing.resume:
            resume_ckpt_path = second_stage_checkpoint_path if cfg.train.stage == "diffusion" else first_stage_checkpoint_path
        else:
            resume_ckpt_path = None
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=resume_ckpt_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=None,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Set to medium for diffusion fine-tuning.
    torch.set_float32_matmul_precision('high')
    train()
