import torch
from torch import nn

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def _count_parameters(model: nn.Module, verbose: bool = True) -> tuple[int, int]:
    """
    Count total and trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.
        verbose (bool): Whether to print the result nicely.

    Returns:
        (n_trainable, n_total): tuple of parameter counts.
    """
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        def fmt(n: int) -> str:
            return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.2f}K" if n >= 1e3 else str(n)
        print(f"Total parameters:     {fmt(n_total)}")
        print(f"Trainable parameters: {fmt(n_trainable)} "
              f"({100 * n_trainable / n_total:.2f}% trainable)")

    return n_trainable, n_total

def _copy_tensor(src_t: torch.Tensor, dst_t: torch.Tensor, name: str) -> bool:
    if tuple(src_t.shape) != tuple(dst_t.shape):
        print(f"[skip] {name}: shape mismatch src {tuple(src_t.shape)} vs dst {tuple(dst_t.shape)}")
        return False
    dst_t.copy_(src_t)
    return True

def _copy_linear(src_lin: nn.Module, dst_lin: nn.Module, name: str) -> bool:
    ok = True
    ok &= _copy_tensor(src_lin.weight, dst_lin.weight, f"{name}.weight")
    if getattr(src_lin, "bias", None) is None:
        if getattr(dst_lin, "bias", None) is not None:
            print(f"[skip] {name}.bias: src has no bias, dst has bias")
            ok = False
    else:
        if getattr(dst_lin, "bias", None) is None:
            print(f"[skip] {name}.bias: src has bias, dst has no bias")
            ok = False
        else:
            ok &= _copy_tensor(src_lin.bias, dst_lin.bias, f"{name}.bias")
    return ok