from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.first_stage.discriminator import DiscriminatorPatchGanCfg
from .model.diffusion_adapter import DiffusionAdapterCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg
from .model.autoencoder import AutoencoderCfg, ColorDecoderCfg, DepthDecoderCfg

# Sentinel for checkpointing.load_first_stage: resolve the Stage-1 VAE checkpoint from the Stage-2
# path instead of spelling it out. Only understands the released layout (see resolve_first_stage).
FIRST_STAGE_AUTO = "auto"
STAGE1_DIR_NAME = "stage1-vae"
STAGE2_DIR_NAME = "stage2-diffusion"


@dataclass
class CheckpointingCfg:
    # Not a path, since it could be something like wandb://... or the "auto" sentinel above.
    load_first_stage: Optional[str]
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    resume: Optional[bool] = True
    load_second_stage: Optional[str] = None  # diffuser/denoiser checkpoint; unused for the vae stage.
    # If set, additionally saves a checkpoint every N steps that is kept forever (never pruned by save_top_k).
    milestone_every_n_steps: Optional[int] = None
    # Extra one-off milestone steps.
    milestone_steps: Optional[list[int]] = None

@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg
    autoencoder: AutoencoderCfg
    color_decoder: ColorDecoderCfg
    depth_decoder: DepthDecoderCfg
    diffuser: DiffusionAdapterCfg | None
    discriminator: Optional[DiscriminatorPatchGanCfg] = None
    
@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    num_nodes: Optional[int] = 1
    devices: Optional[int] = None
    precision: str = "32-true"
    profiler: Literal["simple", "advanced"] | None = None
    accumulate_grad_batches: int = 1  # Only supported for the "diffusion" stage (automatic optimization).

@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def resolve_first_stage(second_stage: Optional[str]) -> str:
    """Infer the Stage-1 VAE checkpoint from the Stage-2 one, for load_first_stage="auto".

    Every Stage 2 checkpoint is paired with exactly one Stage 1 checkpoint, and the released layout
    encodes that pairing in the file names, so users can pass a single path:

        <root>/stage2-diffusion/re10k-video-ft.ckpt  ->  <root>/stage1-vae/re10k.ckpt

    The dataset is the file name up to the first dash (i.e. before -base / -upsample-ft /
    -video-ft). Anything that is not in this layout raises an error.
    """
    if second_stage is None:
        raise ValueError(
            f'checkpointing.load_first_stage="{FIRST_STAGE_AUTO}" needs '
            "checkpointing.load_second_stage to infer from, but it is unset."
        )
    if str(second_stage).startswith("wandb://"):
        raise ValueError(
            f'checkpointing.load_first_stage="{FIRST_STAGE_AUTO}" only works with a local '
            f"checkpoints/ path; load_second_stage is a wandb artifact ({second_stage}). "
            "Pass the Stage-1 checkpoint explicitly."
        )

    second_stage_path = Path(second_stage)
    if second_stage_path.parent.name != STAGE2_DIR_NAME:
        raise ValueError(
            f'checkpointing.load_first_stage="{FIRST_STAGE_AUTO}" expects load_second_stage to sit '
            f"in a {STAGE2_DIR_NAME}/ directory (the released layout), but got {second_stage}. "
            "Pass the Stage-1 checkpoint explicitly."
        )

    dataset = second_stage_path.stem.split("-")[0]
    first_stage_path = second_stage_path.parent.parent / STAGE1_DIR_NAME / f"{dataset}.ckpt"
    if not first_stage_path.is_file():
        raise FileNotFoundError(
            f"Inferred the Stage-1 checkpoint for {second_stage_path.name} as {first_stage_path}, "
            "which does not exist. Download it alongside the Stage-2 checkpoint, or pass "
            "checkpointing.load_first_stage explicitly."
        )
    return str(first_stage_path)


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    typed_cfg = load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )
    if typed_cfg.checkpointing.load_first_stage == FIRST_STAGE_AUTO:
        typed_cfg.checkpointing.load_first_stage = resolve_first_stage(
            typed_cfg.checkpointing.load_second_stage
        )
        print(
            f"[config] load_first_stage={FIRST_STAGE_AUTO} -> "
            f"{typed_cfg.checkpointing.load_first_stage}"
        )
    return typed_cfg
