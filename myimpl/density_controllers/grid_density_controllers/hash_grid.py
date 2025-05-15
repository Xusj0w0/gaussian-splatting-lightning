from dataclasses import dataclass

from .base import (GridGaussianDensityController,
                   GridGaussianDensityControllerImpl)
from .utils import CandidateAnchors, GridFilteringUtils


class HashGridCandidateAnchors(CandidateAnchors):
    def get_scaffold_properties(self, gaussian_model, scatter_mode="max"):
        return {}


class HashGridFilteringUtils(GridFilteringUtils):
    candidate_class_type = HashGridCandidateAnchors


@dataclass
class HashGridGaussianDensityController(GridGaussianDensityController):
    def instantiate(self, *args, **kwargs):
        return HashGridGaussianDensityControllerImpl(self)


class HashGridGaussianDensityControllerImpl(GridGaussianDensityControllerImpl):
    GRID_FILTERING_UTILS = HashGridFilteringUtils

    config: HashGridGaussianDensityController
