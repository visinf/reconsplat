from pathlib import Path

import torch
from torch import Tensor


def blend_latent_pair(prev_latents: Tensor, next_latents: Tensor, overlap: int) -> Tensor:
    """
    Blend two latent sequences along the frame (view) dimension.

    prev_latents: [V1, ...]
    next_latents: [V2, ...]
    overlap: number of overlapping frames between the tail of prev_latents and the head of next_latents.

    Returns a stitched tensor of shape [V1 + V2 - overlap, ...].
    """
    if overlap == 0:
        return torch.cat([prev_latents, next_latents], dim=0)

    assert overlap <= prev_latents.shape[0]
    assert overlap <= next_latents.shape[0]

    prev_non_overlap = prev_latents[:-overlap]
    prev_overlap = prev_latents[-overlap:]
    next_overlap = next_latents[:overlap]
    next_non_overlap = next_latents[overlap:]

    if overlap == 1:
        alpha = torch.tensor([0.5], device=prev_latents.device, dtype=prev_latents.dtype)
    else:
        alpha = torch.linspace(
            0.0, 1.0, steps=overlap,
            device=prev_latents.device,
            dtype=prev_latents.dtype,
        )

    view_shape = [overlap] + [1] * (prev_latents.ndim - 1)
    alpha = alpha.view(*view_shape)

    blended_overlap = (1.0 - alpha) * prev_overlap + alpha * next_overlap

    return torch.cat([prev_non_overlap, blended_overlap, next_non_overlap], dim=0)


def _find_overlap(prev_frame_ids: list[int], next_frame_ids: list[int]) -> int:
    max_possible_overlap = min(len(prev_frame_ids), len(next_frame_ids))
    for o in range(max_possible_overlap, 0, -1):
        if prev_frame_ids[-o:] == next_frame_ids[:o]:
            return o
    return 0


def stitch_chunks(chunks: list[tuple[list[int], Tensor]]) -> tuple[list[int], Tensor]:
    """
    Stitch a sequence of overlapping (frame_ids, latents) chunks for a single scene into one
    continuous latent sequence, blending overlapping frames in latent space.

    chunks: list of (frame_ids, latents), where latents is [V, ...] and len(frame_ids) == V.
    """
    assert len(chunks) > 0, "Need at least one chunk to stitch."

    stitched_frame_ids, stitched_latents = chunks[0]
    stitched_frame_ids = list(stitched_frame_ids)

    for frame_ids, latents in chunks[1:]:
        frame_ids = list(frame_ids)
        overlap = _find_overlap(stitched_frame_ids, frame_ids)
        stitched_latents = blend_latent_pair(stitched_latents, latents, overlap)
        stitched_frame_ids = (
            stitched_frame_ids[:-overlap] + frame_ids if overlap > 0 else stitched_frame_ids + frame_ids
        )

    return stitched_frame_ids, stitched_latents


def load_scene_windows(scene_folder: Path) -> list[tuple[list[int], Tensor, Tensor, Tensor, Tensor]]:
    """
    Load latent windows previously saved to disk.

    Returns a list of (frame_ids, color_latents, depth_latents, near, far) per saved window, sorted by filename.
    """
    window_paths = sorted(scene_folder.glob("latent_windows/*.torch"))
    if not window_paths:
        raise ValueError(f"No latent windows found in {scene_folder}")
    windows = [torch.load(p, map_location="cpu") for p in window_paths]
    return [
        (list(w["frame_ids"]), w["color_latents"], w["depth_latents"], w["near"], w["far"])
        for w in windows
    ]
