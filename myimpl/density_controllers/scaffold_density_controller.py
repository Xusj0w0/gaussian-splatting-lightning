from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from einops import repeat
from torch_scatter import scatter_max

from myimpl.density_controllers.octree_density_controller import (
    OctreeDensityController, OctreeDensityControllerImpl)
from myimpl.models.scaffold_gaussian import ScaffoldLoDGaussianModel


@dataclass
class ScaffoldDensityController(OctreeDensityController):
    def instantiate(self, *args, **kwargs):
        return ScffoldDensityControllerImpl(self)


class ScffoldDensityControllerImpl(OctreeDensityControllerImpl):
    config: ScaffoldDensityController

    def _get_new_properties(
        self, candidate_infos: Tuple[torch.Tensor], gaussian_model: ScaffoldLoDGaussianModel, cur_size: float
    ):
        properties, num_anchors = super()._get_new_properties(candidate_infos, gaussian_model, cur_size)
        _, _, grad_mask, unique_indices, filter_mask = candidate_infos

        new_anchor_features = repeat(
            gaussian_model.get_anchor_features, "n c -> (n o) c", o=gaussian_model.config.n_offsets
        )
        new_anchor_features = new_anchor_features[: len(grad_mask)][grad_mask]  # maybe has add anchors
        new_anchor_features = scatter_max(
            new_anchor_features, unique_indices.unsqueeze(1).expand(-1, new_anchor_features.size(-1)), dim=0
        )[0][filter_mask]
        properties.update({"anchor_features": new_anchor_features})
        return properties, num_anchors
