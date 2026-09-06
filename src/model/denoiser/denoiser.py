from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Tuple
import torch
from jaxtyping import Float, Int64
from torch import nn, Tensor



T = TypeVar("T")


class Denoiser(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self, 
        cfg: T
    ) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        latents: Float[Tensor, "batch view _ height width"],
        timestep: Int64[Tensor, "batch"],
        cond_state: Optional[Tensor]=None,
        upscale_factor: Optional[float]=8.0,
        extrinsics: Optional[Float[Tensor, "batch view i i"]] = None,
        intrinsics: Optional[Float[Tensor, "batch view j j"]] = None,
        unconditional: bool = False
    ) -> (
        Float[Tensor, "batch view _ height width"]
        | Tuple[
            Float[Tensor, "batch view _ height width"],
            dict[int, Float[Tensor, "..."]]
        ]
    ):
        pass
