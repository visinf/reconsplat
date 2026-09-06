from dataclasses import dataclass
from typing import Optional, Literal, Tuple

@dataclass
class ProcessorConfig:
    dataset_name: Literal["re10k", "dl3dv"] = "re10k"
    from_chunks: bool = False
    save_back_to_chunks: bool = False 
    save_cameras_to_npz: bool = False
    save_depth_as_npz: bool = True
    image_load_resolution: Tuple[int, int] = (252, 448)
    skip_already_processed_chunks: bool = True
    save_progress: bool = False
    skip_cache_file: str = "processed_all.txt"
    progress_dir: str = "progress_last"
    errors_dir: str = "errors_last" 

@dataclass
class StructureConfig:
    feed_forward: bool = True
    load_from_url: bool = True
    model_path: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_fixed_resolution: Tuple[int, int] = (252, 448)
    patch_size: int = 14
    save_points_3d: bool = True
    sample_points_3d: bool = True
    max_points_for_colmap: int = 100000
    conf_thres_value: float = 5.0
    save_intermediate_sparse: bool = True
    save_intermediate_dense: bool = True
    save_images: bool = True
    save_video_depth: bool = False

@dataclass
class VideoDepthConfig:
    model_dir: str 
    encoder: str = "vitl"

@dataclass
class PipelineConfig:
    structure: StructureConfig
    video_depth: VideoDepthConfig
    processor: ProcessorConfig
    load_dir: str
    out_dir: str
    align_with_video_depth: bool = True
    save_recon_after_align: bool = False
    save_aligned_depth_video: bool = False
    scale_predictions_to_orginal_res: bool = True

@dataclass
class PseudoLabelsCfg:
    seed: Optional[int] = None
    pipeline: PipelineConfig = None



