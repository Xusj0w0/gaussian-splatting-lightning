import os.path as osp
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from einops import repeat
from lightning import LightningModule

from internal.utils.partitioning_utils import (MinMaxBoundingBox,
                                               PartitionCoordinates)
from myimpl.models.grid_gaussians import GridGaussianModel

from .base import (GridGaussianDensityController,
                   GridGaussianDensityControllerImpl)
from .utils import CandidateAnchors, GridFilteringUtils

__all__ = [
    "PartitionableGridGaussianDensityController",
    "PartitionableGridGaussianDensityControllerImpl",
]


class PartitionInfo:
    def __init__(self, partition_info_path: str, partition_name: str):
        partition_info = torch.load(partition_info_path, map_location="cpu")

        # get manhattan transform
        self.manhattan_trans: torch.Tensor = (
            partition_info["extra_data"]["rotation_transform"] if partition_info["extra_data"] is not None else None
        )

        # get partition bounding box
        coords = PartitionCoordinates(**partition_info["partition_coordinates"])

        partition_bbox = coords.get_bounding_boxes()
        scene_bbox = (torch.min(partition_bbox.min, dim=0).values, torch.max(partition_bbox.max, dim=0).values)

        for idx in range(len(coords)):
            if coords.get_str_id(idx) == partition_name:
                bounding_box = partition_bbox[idx]
                # enlarge
                self.bounding_box = MinMaxBoundingBox(
                    min=bounding_box.min - 0.1 * (bounding_box.max - bounding_box.min),
                    max=bounding_box.max + 0.1 * (bounding_box.max - bounding_box.min),
                )

                # update
                if torch.isclose(bounding_box.min[0], scene_bbox[0][0], atol=1e-4):
                    # x == scene bbox x min -> bbox.x_min = -inf
                    self.bounding_box.min[0] = -torch.inf
                    print("xmin updated to -inf")
                if torch.isclose(bounding_box.min[1], scene_bbox[0][1], atol=1e-4):
                    # y == scene bbox y min -> bbox.y_min = -inf
                    self.bounding_box.min[1] = -torch.inf
                    print("ymin updated to -inf")
                if torch.isclose(bounding_box.max[0], scene_bbox[1][0], atol=1e-4):
                    # x == scene bbox x max -> bbox.x_max = inf
                    self.bounding_box.max[0] = torch.inf
                    print("xmax updated to inf")
                if torch.isclose(bounding_box.max[1], scene_bbox[1][1], atol=1e-4):
                    # x == scene bbox x max -> bbox.x_max = inf
                    self.bounding_box.max[1] = torch.inf
                    print("ymax updated to inf")

                break

    def is_in_partition(self, coordinates: torch.Tensor):
        if self.manhattan_trans.device != coordinates.device:
            self.manhattan_trans = self.manhattan_trans.to(coordinates.device)
            self.bounding_box = MinMaxBoundingBox(
                min=self.bounding_box.min.to(coordinates.device),
                max=self.bounding_box.max.to(coordinates.device),
            )

        coordinates = coordinates @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        mask = torch.logical_and(
            torch.prod(coordinates[:, :2] > self.bounding_box.min[:2], dim=-1),
            torch.prod(coordinates[:, :2] < self.bounding_box.max[:2], dim=-1),
        )
        return mask


@dataclass
class PartitionableCandidateAnchors(CandidateAnchors):
    def get_all_properties(
        self,
        gaussian_model: GridGaussianModel,
        voxel_size: float,
        scatter_mode: Literal["max", "mean"] = "max",
        partition_info: PartitionInfo = None,
    ):
        property_dict = super().get_all_properties(gaussian_model, voxel_size, scatter_mode)
        if partition_info is not None:
            mask = partition_info.is_in_partition(self.anchors)
            _property_dict = {k: v[mask] for k, v in property_dict.items()}
            property_dict = _property_dict
        return property_dict


class PartitionableFilteringUtils(GridFilteringUtils):
    candidate_class_type = PartitionableCandidateAnchors


@dataclass
class PartitionableGridGaussianDensityController(GridGaussianDensityController):
    densify_in_partition: bool = True
    prune_in_partition: bool = True

    def instantiate(self, *args, **kwargs):
        return PartitionableGridGaussianDensityControllerImpl(self)


class PartitionableGridGaussianDensityControllerImpl(GridGaussianDensityControllerImpl):
    GRID_FILTERING_UTILS = PartitionableFilteringUtils

    config: PartitionableGridGaussianDensityController

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    @property
    def density_state_names(self):
        return self._density_state_names

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        partition_info_path = osp.join(pl_module.hparams["output_path"], "../..", "partition_infos/partitions.pt")
        if osp.exists(partition_info_path):
            partition_name = osp.basename(
                pl_module.hparams["output_path"]
            )  # TODO: find attribute `name` from `pl_module`
            partition_info = PartitionInfo(partition_info_path, partition_name)
            if hasattr(partition_info, "bounding_box"):
                self.partition_info = partition_info

    def _densify_and_prune(self, gaussian_model: GridGaussianModel, optimizers: List):
        n_offsets = gaussian_model.n_offsets

        grads_norm = self._primitive_gradient_accum / self._primitive_denom
        grads_norm[grads_norm.isnan()] = 0.0
        # grads_norm = torch.norm(grads, dim=-1)
        denom_thresh = self.config.densification_interval * self.config.success_threshold * 0.5
        primitive_mask = self._primitive_denom > denom_thresh

        # densify anchors
        if getattr(gaussian_model, "get_levels", None) is not None and gaussian_model.get_levels.shape[0] > 0:
            self.densify_anchors_multilevel(grads_norm, primitive_mask, gaussian_model, optimizers)
        else:
            self.densify_anchors_paperversion(grads_norm, primitive_mask, gaussian_model, optimizers)

        # enlarge buffers
        num_anchors = gaussian_model.n_anchors - self._anchor_denom.shape[0]
        self._densify_buffers(num_anchors, num_anchors * n_offsets, primitive_mask)
        torch.cuda.empty_cache()

        # prune anchors
        anchor_denom_thresh = self.config.densification_interval * self.config.success_threshold
        opacity_mask = self._anchor_opacity_accum < self.config.cull_opacity_threshold * self._anchor_denom
        denom_mask = self._anchor_denom > anchor_denom_thresh
        remove_mask = torch.logical_and(denom_mask, opacity_mask)
        if hasattr(self, "partition_info") and self.config.prune_in_partition:
            is_in_partition = self.partition_info.is_in_partition(gaussian_model.get_xyz)
            remove_mask = torch.logical_and(remove_mask, is_in_partition)
        keep_mask = ~remove_mask

        # re-accumulate: set denom and opacity_accum of anchors that denom > denom_thresh to 0
        # prune anchors that denom > denom_thresh and opacity_accum < opacity_thresh
        self._prune_anchors(keep_mask, gaussian_model, optimizers)

        # prune buffers
        self._prune_buffers(denom_mask, keep_mask, n_offsets)
        torch.cuda.empty_cache()

        # register buffer
        self.register_density_states()
