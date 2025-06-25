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

    def get_features_for_render(self, pc, properties, visibility_filter):
        xyz, *_ = properties
        positions = xyz[visibility_filter]
        features = pc.compute_hash_features(positions)
        return features
