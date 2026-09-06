import torch

class ConstantGuider:
    def __init__(self, scale: float):
        """
        scale: classifier-free guidance scale, e.g. 3.0
        """
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: float, **kwargs) -> torch.Tensor:
        """
        x: tensor with unconditional/conditional stacked in batch dim,
           shape [2 * B, ...] (e.g. [2B, T, C, H, W] or [2B, V, C, H, W])
        sigma: noise level (ignored here, but kept for API compatibility)
        """
        x_u, x_c = x.chunk(2)  # [B, ...] each
        # same CFG scale for all frames/views, broadcasting over remaining dims
        return x_u + self.scale * (x_c - x_u)
