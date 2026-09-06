import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Literal

import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from tqdm import tqdm

from ..geometry.epipolar_lines import project_rays
from ..geometry.projection import get_world_rays, sample_image_grid
from ..misc.image_io import save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat


@dataclass
class EvaluationIndexGeneratorCfg:
    num_context_pairs_per_scene: int
    num_context_views: int
    num_target_views: int
    min_context_overlap: float
    max_context_overlap: float
    min_context_distance: int
    max_context_distance: int
    max_target_distance: int   # will be ignored if intra_context
    intra_context: bool
    output_path: Path
    save_previews: bool
    seed: int
    view_selection_ratio: float = 1.0
    extra_views_sampling_strategy: Optional[Literal["random", "farthest_point", "equal"]] = "random"

@dataclass
class IndexEntry:
    context: tuple[int, ...] | torch.Tensor
    target: tuple[int, ...] | torch.Tensor 
    target_quantify: Optional[tuple[int, ...] | None | torch.Tensor] = None

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    barycenter = torch.sum((xyz), 1)
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]

    for i in range(npoint):
        # print("-------------------------------------------------------")
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids

class EvaluationIndexGenerator(LightningModule):
    generator: torch.Generator
    cfg: EvaluationIndexGeneratorCfg
    index: dict[str, list[IndexEntry]]

    def __init__(self, cfg: EvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}
        print(f'Running with view_selection_ratio = {self.cfg.view_selection_ratio}')
        print(f'Running with extra_views_sampling_strategy = {self.cfg.extra_views_sampling_strategy}')

    def test_step(self, batch, batch_idx):
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        extrinsics = batch["target"]["extrinsics"][0]
        intrinsics = batch["target"]["intrinsics"][0]
        scene = batch["scene"][0]

        views = []
        
        if self.cfg.view_selection_ratio < 1.0:
            v = int(v * self.cfg.view_selection_ratio)
        
        context_indices = torch.randperm(v, generator=self.generator)
        for context_index in tqdm(context_indices, "Finding context pair"):
            xy, _ = sample_image_grid((h, w), self.device)
            context_origins, context_directions = get_world_rays(
                rearrange(xy, "h w xy -> (h w) xy"),
                extrinsics[context_index],
                intrinsics[context_index],
            )

            # Step away from context view until the minimum overlap threshold is met.
            valid_indices = []
            for step in (1, -1):
                min_distance = self.cfg.min_context_distance
                max_distance = self.cfg.max_context_distance
                current_index = context_index + step * min_distance
                
                while 0 <= current_index.item() < v:
                    # Compute overlap.
                    current_origins, current_directions = get_world_rays(
                        rearrange(xy, "h w xy -> (h w) xy"),
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_current = project_rays(
                        context_origins,
                        context_directions,
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_context = project_rays(
                        current_origins,
                        current_directions,
                        extrinsics[context_index],
                        intrinsics[context_index],
                    )
                    overlap_a = projection_onto_context["overlaps_image"].float().mean()
                    overlap_b = projection_onto_current["overlaps_image"].float().mean()

                    overlap = min(overlap_a, overlap_b)
                    delta = (current_index - context_index).abs()

                    min_overlap = self.cfg.min_context_overlap
                    max_overlap = self.cfg.max_context_overlap
                    if min_overlap <= overlap <= max_overlap:
                        valid_indices.append(
                            (current_index.item(), overlap_a, overlap_b)
                        )

                    # Stop once the camera has panned away too much.
                    if overlap < min_overlap or delta > max_distance:
                        break

                    current_index += step

            if valid_indices:
                # Pick a random valid view. Index the resulting views.
                num_options = len(valid_indices)
                chosen = torch.randint(
                    0, num_options, size=tuple(), generator=self.generator
                )
                chosen, overlap_a, overlap_b = valid_indices[chosen]

                context_left = min(chosen, context_index.item())
                context_right = max(chosen, context_index.item())
                delta = context_right - context_left

                if self.cfg.intra_context:
                    target_views = torch.arange(context_left, context_right + 1)
                else:
                    target_views = torch.cat((
                        torch.arange(
                            max(context_left-self.cfg.max_target_distance, 0),
                            context_left
                        ),
                        torch.arange(
                            context_right + 1,
                            min(context_right+self.cfg.max_target_distance+1, v)
                        )
                    ))

                if len(target_views) < self.cfg.num_target_views:
                    continue

                rand_idx = torch.randperm(target_views.shape[0], generator=self.generator)
                target_views = target_views[rand_idx][:self.cfg.num_target_views]
                target = tuple(torch.sort(target_views).values.tolist())

                # Pick extra context views between the left and right ones.
                if self.cfg.num_context_views > 2:
                    num_extra_views = self.cfg.num_context_views - 2
                    extra_views = []
                    if self.cfg.extra_views_sampling_strategy == 'random':
                        while len(set(extra_views)) != num_extra_views:
                            extra_views = torch.randint(
                                context_left + 1,
                                context_right,
                                (num_extra_views,),
                            ).tolist()
                    elif self.cfg.extra_views_sampling_strategy == 'farthest_point':
                        context_bounded_index = torch.arange(context_left, context_right + 1)
                        candidate_views_position = extrinsics[context_bounded_index, :3, -1].unsqueeze(0)
                        index_context_local = farthest_point_sample(
                            candidate_views_position, self.cfg.num_context_views
                        ).squeeze(0).cpu()
                        # remap context index back to global scene based index
                        index_context = context_bounded_index[index_context_local]
                        context_left = index_context[0].item()
                        context_right = index_context[-1].item()
                        extra_views = index_context[1:-1].tolist()
                    elif self.cfg.extra_views_sampling_strategy == 'equal':
                        pass
                else:
                    extra_views = []

                views.append(IndexEntry(
                    context=(context_left, *extra_views, context_right),
                    target=target,
                ))

                # Optionally, save a preview.
                if self.cfg.save_previews:
                    preview_path = self.cfg.output_path / "previews"
                    preview_path.mkdir(exist_ok=True, parents=True)
                    a = batch["target"]["image"][0, chosen]
                    a = add_label(a, f"Overlap: {overlap_a * 100:.1f}%")
                    b = batch["target"]["image"][0, context_index]
                    b = add_label(b, f"Overlap: {overlap_b * 100:.1f}%")
                    vis = add_border(add_border(hcat(a, b)), 1, 0)
                    vis = add_label(vis, f"Distance: {delta} frames")
                    save_image(add_border(vis), preview_path / f"{scene}.png")
                
                if len(views) == self.cfg.num_context_pairs_per_scene:
                    break
        
        self.index[scene] = views

    def save_index(self) -> None:
        self.cfg.output_path.mkdir(exist_ok=True, parents=True)
        with (self.cfg.output_path / "evaluation_index.json").open("w") as f:
            json.dump(
                {k: [asdict(v) for v in views] for k, views in self.index.items()}, f
            )