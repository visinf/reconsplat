from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg

# Base seed for the chunk-order permutation the IterableDataset loaders apply before sharding
# chunks across the (rank x worker) grid. Deliberately a fixed constant rather than anything
# rank- or worker-derived: every loader process must draw the *identical* permutation, otherwise
# the shard stops being a partition (chunks get both duplicated and dropped) and the per-rank
# stream lengths diverge, which deadlocks DDP at the epoch boundary. See _shard_chunks.
SHARD_SHUFFLE_SEED = 20260729


@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
