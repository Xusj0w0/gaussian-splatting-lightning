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
from myimpl.renderers.grid_renderer import GridRendererUtils
from myimpl.renderers.hash_grid_renderer import (
    HashGridGaussianRenderer, HashGridGaussianRendererModule)
from myimpl.utils.cameras import InstantiatedCameras
from myimpl.utils.loss_utils import MultiView


@dataclass
class HashGridAssistedGaussianRenderer(HashGridGaussianRenderer):
    """HashGridGaussianRenderer is a renderer that uses hash grid encoding."""

    def instantiate(self, *args, **kwargs):
        return HashGridAssistedGaussianRendererModule(self)


class HashGridAssistedGaussianRendererModule(HashGridGaussianRendererModule):
    """HashGridGaussianRendererModule is a module that uses hash grid encoding."""

    config: HashGridAssistedGaussianRenderer

    def render_feature(
        self,
        properties_list: List[Tuple[torch.Tensor, ...]],
        viewpoint_camera: Cameras,
        pc,
        scaling_modifier=1.0,
        **kwargs,
    ):
        render_feature, _, alpha = super().render_feature(
            properties_list, viewpoint_camera, pc, scaling_modifier, **kwargs
        )
        aligned_features = None
        adapter = getattr(pc, "get_feature_adapter_mlp", None)
        if adapter is not None:
            aligned_features = adapter(render_feature)

        return render_feature, aligned_features, alpha
