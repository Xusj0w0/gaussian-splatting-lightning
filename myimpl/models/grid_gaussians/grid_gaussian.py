from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import lightning
import torch
import torch.nn as nn

from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.network_factory import NetworkFactory

from .base import (GridGaussianBase, GridGaussianModelBase,
                   GridOptimizationConfigBase)
from .utils import GridFactory

__all__ = ["GridGaussian", "GridGaussianModel", "GridOptimizationConfig"]


@dataclass
class GridOptimizationConfig(GridOptimizationConfigBase):
    pass


@dataclass
class GridGaussian(GridGaussianBase):
    update_depth: int = 3

    update_init_factor: int = 16

    update_hierachy_factor: int = 4

    optimization: GridOptimizationConfig = field(default_factory=GridOptimizationConfig)

    def instantiate(self, *args, **kwargs):
        return GridGaussianModel(self)


class GridGaussianModel(GridGaussianModelBase):
    config: GridGaussian
