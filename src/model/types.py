from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    color_feature_harmonics: Float[Tensor, "batch gaussian channels d_feature_sh"] | None = (None)
    color_features: Float[Tensor, "batch gaussian channels"] | None = (None)
    geometry_features: Float[Tensor, "batch gaussian channels"] | None = (None)
