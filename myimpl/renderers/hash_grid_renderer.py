import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import lightning
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from internal.cameras.cameras import Camera, Cameras
from myimpl.model_components.decoupled_appearance_model import \
    DecoupledAppearanceModelConfig
from myimpl.models.grid_gaussians import (GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)
from myimpl.renderers.grid_renderer import (GridGaussianRenderer,
                                            GridGaussianRendererModule,
                                            GridRendererUtils)
from myimpl.utils.cameras import InstantiatedCameras
from myimpl.utils.loss_utils import MultiView


@dataclass
class HashGridGaussianRenderer(GridGaussianRenderer):
    """HashGridGaussianRenderer is a renderer that uses hash grid encoding."""

    def instantiate(self, *args, **kwargs):
        return HashGridGaussianRendererModule(self)


class HashGridGaussianRendererModule(GridGaussianRendererModule):
    """HashGridGaussianRendererModule is a module that uses hash grid encoding."""

    config: HashGridGaussianRenderer

    def render_feature(
        self,
        properties_list: List[Tuple[torch.Tensor, ...]],
        viewpoint_camera: Cameras,
        pc: GridGaussianModel,
        scaling_modifier=1.0,
        **kwargs,
    ):
        projections_list, isects_list, visibility_filter, preprocessed_camera = (
            GridRendererUtils.project_to_pixels_loop(
                self,
                properties_list,
                viewpoint_camera,
                scaling_modifier=scaling_modifier,
                return_preprocessed_cam=True,
                **kwargs,
            )
        )
        projections = GridRendererUtils.concatenate_projections(projections_list, isects_list)
        means2d, *_ = projections

        input_features = means2d.new_zeros((0, pc.config.hash_feature_grid.out_dim))
        input_opacities = means2d.new_zeros((0,))
        for cam_id in range(len(viewpoint_camera)):
            _xyz, _, _, _, _opacities, _, _, *_ = properties_list[cam_id]
            _visibility_filter = visibility_filter[cam_id]
            positions = _xyz[_visibility_filter].clone().detach()
            features = pc.compute_hash_features(positions)

            input_opacities = torch.cat([input_opacities, _opacities[_visibility_filter]], dim=0)
            input_features = torch.cat([input_features, features], dim=0)

        render_feature, alpha = GridRendererUtils.rasterize_cat_projections(
            preprocessed_camera=preprocessed_camera,
            projections=projections,
            properties=(input_features, input_opacities),
            bg_color=means2d.new_zeros((len(viewpoint_camera), input_features.shape[-1])),
            tile_size=self.config.block_size,
        )

        return render_feature, None, alpha
