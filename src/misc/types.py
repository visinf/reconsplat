from dataclasses import dataclass
from jaxtyping import Float, Bool
from torch import Tensor

@dataclass
class DiffuserOutput:
    pred_color: Float[Tensor, "batch view 4 h w"]
    pred_depth: Float[Tensor, "batch view 4 h w"] | None
    valid_mask_depth: Bool[Tensor, "batch view 1 h w"] | None
    gt_color: Float[Tensor, "batch view 4 h w"]
    gt_depth: Float[Tensor, "batch view 4 h w"]