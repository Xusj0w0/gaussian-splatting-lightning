from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import repeat
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from internal.cameras.cameras import Cameras
from internal.density_controllers.density_controller import \
    Utils as OptimizerManipulator
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from myimpl.models.grid_gaussians import (GridFactory, GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)


@dataclass
class CandidateAnchors:
    anchors: torch.Tensor
    """filtered anchors"""

    levels: torch.Tensor
    """filtered levels"""

    grad_mask: torch.Tensor
    """filter primivites by grad_mask"""

    unique_indices: torch.Tensor
    """convert to grids and select unique grids"""

    keep_mask: torch.Tensor
    """remove existing anchors & filter by predicted level"""

    @property
    def n_anchors(self) -> int:
        return self.anchors.shape[0]

    def get_basic_properties(self, gaussian_model: GridGaussianModel, voxel_size: float):
        scales = gaussian_model.scale_inverse_activation(
            gaussian_model.get_scalings.new_ones((self.n_anchors, 6)) * voxel_size
        )
        offsets = gaussian_model.get_anchors.new_zeros((self.n_anchors, gaussian_model.n_offsets, 3))
        rotations = gaussian_model.get_anchors.new_zeros((self.n_anchors, 4))
        rotations[..., 0] = 1.0
        return {"means": self.anchors, "scales": scales, "offsets": offsets, "rotations": rotations}

    def get_lod_grid_properties(self, gaussian_model: LoDGridGaussianModel):
        extra_levels = gaussian_model.get_extra_levels.new_zeros((self.n_anchors,))
        return {"levels": self.levels, "extra_levels": extra_levels}

    def get_scaffold_properties(
        self, gaussian_model: ScaffoldGaussianModelMixin, scatter_mode: Literal["max", "mean"] = "max"
    ):
        # original implementation
        # TODO: CUDA OOM!
        # anchor_features = repeat(gaussian_model.get_anchor_features, "n c -> (n o) c", o=gaussian_model.n_offsets)
        # # if anchors are added, shape of anchor_features may dismatch grad_mask
        # anchor_features = anchor_features[: len(self.grad_mask)][self.grad_mask]

        # # select value of anchor features among primitives that convert to same grid
        # if scatter_mode == "max":
        #     anchor_features = scatter_max(anchor_features, self.unique_indices.unsqueeze(1).expand(-1, anchor_features.shape[-1]), dim=0)[0]  # fmt: skip
        # elif scatter_mode == "mean":
        #     anchor_features = scatter_mean(anchor_features, self.unique_indices.unsqueeze(1).expand(-1, anchor_features.shape[-1]), dim=0)  # fmt: skip
        # else:
        #     raise ValueError(f"scatter_mode {scatter_mode} not supported")
        # anchor_features = anchor_features[self.keep_mask]

        keep_indices = torch.nonzero(self.keep_mask, as_tuple=True)[0]
        is_keep = self.unique_indices.unsqueeze(1) == keep_indices.unsqueeze(0)
        keep_mask, keep_idx_mapping = torch.nonzero(is_keep, as_tuple=True)

        indices = (torch.nonzero(self.grad_mask, as_tuple=True)[0] / gaussian_model.n_offsets).long()[keep_mask]
        anchor_features = gaussian_model.get_anchor_features[indices]
        feat_dim = anchor_features.shape[-1]

        if scatter_mode == "max":
            anchor_features = scatter_max(anchor_features, keep_idx_mapping.unsqueeze(1).expand(-1, feat_dim), dim=0)[0]
        elif scatter_mode == "mean":
            anchor_features = scatter_mean(anchor_features, keep_idx_mapping.unsqueeze(1).expand(-1, feat_dim), dim=0)
        else:
            raise ValueError(f"scatter_mode {scatter_mode} not supported")
        return {"anchor_features": anchor_features}  # , "opacities": opacities}

    def get_all_properties(
        self, gaussian_model: GridGaussianModel, voxel_size, scatter_mode: Literal["max", "mean"] = "max"
    ):
        property_dict = self.get_basic_properties(gaussian_model, voxel_size)
        if getattr(gaussian_model, "get_levels", None) is not None and gaussian_model.get_levels.shape[0] > 0:
            property_dict.update(self.get_lod_grid_properties(gaussian_model))
        if getattr(gaussian_model, "gaussian_mlps", None) is not None:
            property_dict.update(self.get_scaffold_properties(gaussian_model, scatter_mode))
        # TODO: explicit model
        return property_dict


class GridFilteringUtils:
    candidate_class_type = CandidateAnchors

    @staticmethod
    def filter_exsiting_grids(
        candidate_grids: torch.Tensor, existing_grids: torch.Tensor, overlap: int = 1, use_chunk=True
    ):
        assert overlap > 0
        count = candidate_grids.new_zeros((candidate_grids.shape[0],), dtype=torch.int)
        if use_chunk:
            chunk_size = 4096
            max_iters = existing_grids.shape[0] // chunk_size + (1 if existing_grids.shape[0] % chunk_size != 0 else 0)
            for i in range(max_iters):
                cur_existing_grids = existing_grids[i * chunk_size : (i + 1) * chunk_size, :]
                matches = (candidate_grids.unsqueeze(1) == cur_existing_grids).all(-1)
                count += matches.sum(-1)
        else:
            count = (candidate_grids.unsqueeze(1) == cur_existing_grids).all(-1).sum(-1)

        return count < overlap

    @classmethod
    def filter_primitives(
        cls,
        gaussian_model: GridGaussianModel,
        all_primitives: torch.Tensor,  # Avoid duplicate calculations
        grad_mask: torch.Tensor,
        overlap: int = 1,
    ):
        # filter by grad mask
        candidate_primitives = all_primitives.view(-1, 3)[grad_mask]

        # convert to grids and select unique grids
        # `unique_indices` is a (candidate_grids.shape[0], ) long tensor
        # same grids are marked with same value
        existing_grids = gaussian_model.xyz2grid(gaussian_model.get_anchors, gaussian_model.voxel_size)
        candidate_grids = gaussian_model.xyz2grid(candidate_primitives, gaussian_model.voxel_size)
        candidate_grids, unique_indices = torch.unique(candidate_grids, return_inverse=True, dim=0)

        # initial values
        filtered_anchors = candidate_primitives.new_zeros((0, 3))
        keep_mask = existing_grids.new_zeros((candidate_grids.shape[0],), dtype=torch.bool)

        if overlap < 0:
            keep_mask = existing_grids.new_ones((candidate_grids.shape[0],), dtype=torch.bool)
            filtered_anchors = gaussian_model.grid2xyz(candidate_grids, gaussian_model.voxel_size)
        else:
            keep_mask = cls.filter_exsiting_grids(candidate_grids, existing_grids, overlap=overlap)
            filtered_anchors = gaussian_model.grid2xyz(candidate_grids[keep_mask], gaussian_model.voxel_size)

        return cls.candidate_class_type(
            anchors=filtered_anchors,
            levels=None,
            grad_mask=grad_mask,
            unique_indices=unique_indices,
            keep_mask=keep_mask,
        )

    @classmethod
    def filter_primitives_paperversion(
        cls,
        gaussian_model: GridGaussianModel,
        all_primitives: torch.Tensor,  # Avoid duplicate calculations
        grad_mask: torch.Tensor,
        voxel_size: float,
        overlap: int = 1,
    ):
        # filter by grad mask
        candidate_primitives = all_primitives.view(-1, 3)[grad_mask]

        # convert to grids and select unique grids
        existing_grids = gaussian_model.xyz2grid(gaussian_model.get_anchors, voxel_size)
        candidate_grids = gaussian_model.xyz2grid(candidate_primitives, voxel_size)
        candidate_grids, unique_indices = torch.unique(candidate_grids, return_inverse=True, dim=0)

        # initial values
        filtered_anchors = candidate_primitives.new_zeros((0, 3))
        keep_mask = existing_grids.new_zeros((candidate_grids.shape[0],), dtype=torch.bool)

        if overlap < 0:
            keep_mask = existing_grids.new_ones((candidate_grids.shape[0],), dtype=torch.bool)
            filtered_anchors = gaussian_model.grid2xyz(candidate_grids, gaussian_model.voxel_size)
        else:
            keep_mask = cls.filter_exsiting_grids(candidate_grids, existing_grids, overlap=overlap)
            filtered_anchors = gaussian_model.grid2xyz(candidate_grids[keep_mask], gaussian_model.voxel_size)

        return cls.candidate_class_type(
            anchors=filtered_anchors,
            levels=None,
            grad_mask=grad_mask,
            unique_indices=unique_indices,
            keep_mask=keep_mask,
        )

    @classmethod
    def filter_lod_primitives(
        cls,
        gaussian_model: LoDGridGaussianModel,
        all_primitives: torch.Tensor,  # Avoid duplicate calculations
        grad_mask: torch.Tensor,
        res_level: torch.Tensor,
        level_mask: torch.Tensor,  # Avoid duplicate calculations
        cam_infos: torch.Tensor,
        overlap: int = 1,
        is_next_level: bool = False,
    ):
        """
        1. primitives are filtered by grad mask (grad_mask)
        2. convert to grids and select unique grids (unique_indices)
        3. filter by existing anchors (if overlap);
        4. filter by predicted level according to distances to train cameras (step 3/4 -> keep_mask)
        """
        voxel_size = gaussian_model.voxel_size / (float(gaussian_model.config.fork) ** res_level)

        # filter by grad mask
        candidate_primitives = all_primitives.view(-1, 3)[grad_mask]

        # convert to grids and select unique grids
        # `unique_indices` is a (candidate_grids.shape[0], ) long tensor
        # same grids are marked with same value
        existing_grids = gaussian_model.xyz2grid(gaussian_model.get_anchors[level_mask], voxel_size)
        candidate_grids = gaussian_model.xyz2grid(candidate_primitives, voxel_size)
        candidate_grids, unique_indices = torch.unique(candidate_grids, return_inverse=True, dim=0)

        # initial values
        filtered_anchors = candidate_primitives.new_zeros((0, 3))
        filtered_levels = gaussian_model.get_levels.new_zeros((0,))
        keep_mask = existing_grids.new_zeros((candidate_grids.shape[0],), dtype=torch.bool)

        # if is current level, then directly filter by existing anchors and weed out by cameras
        # if is next level, execute filtering after activate_level == max_level
        # and current level shouldn't exceed max_level
        if not is_next_level or (
            gaussian_model.activate_level >= gaussian_model.max_level and res_level < gaussian_model.max_level
        ):
            if candidate_grids.shape[0] > 0:
                # don't filter by existing anchors
                if overlap < 0:
                    keep_mask = existing_grids.new_ones((candidate_grids.shape[0],), dtype=torch.bool)
                    candidate_anchors = gaussian_model.grid2xyz(candidate_grids, voxel_size)
                else:
                    keep_mask = cls.filter_exsiting_grids(candidate_grids, existing_grids, overlap=overlap)
                    candidate_anchors = gaussian_model.grid2xyz(candidate_grids[keep_mask], voxel_size)

                candidate_levels = gaussian_model.get_levels.new_ones((candidate_anchors.shape[0],)) * res_level
                weed_keep_mask = GridFactory.weed_out_mask_by_level(
                    candidate_anchors,
                    candidate_levels,
                    gaussian_model.visibility_threshold,
                    cam_infos=cam_infos,
                    predict_level_fn=gaussian_model.predict_level,
                    int_level_fn=lambda x: gaussian_model.map_to_int_level(x, gaussian_model.max_level)[0],
                )
                keep_mask[keep_mask.clone()] = weed_keep_mask
                filtered_anchors, filtered_levels = candidate_anchors[weed_keep_mask], candidate_levels[weed_keep_mask]

        return cls.candidate_class_type(
            anchors=filtered_anchors,
            levels=filtered_levels,
            grad_mask=grad_mask,
            unique_indices=unique_indices,
            keep_mask=keep_mask,
        )
