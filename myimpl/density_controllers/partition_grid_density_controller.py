import os.path as osp
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from einops import repeat
from lightning import LightningModule
from torch_scatter import scatter_max, scatter_mean

from internal.cameras.cameras import Cameras
from internal.density_controllers.density_controller import \
    Utils as OptimizerManipulator
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from internal.utils.partitioning_utils import (MinMaxBoundingBox,
                                               PartitionCoordinates)
from myimpl.density_controllers.grid_density_controller import (
    GridGaussianDensityController, GridGaussianDensityControllerImpl)
from myimpl.models.grid_gaussians import (GridFactory, GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)

__all__ = [
    "PartitionGridGaussianDensityController",
    "PartitionGridGaussianDensityControllerImpl",
    "GridFilteringUtils",
    "CandidateAnchors",
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
class PartitionGridGaussianDensityController(GridGaussianDensityController):
    densify_in_partition: bool = True
    prune_in_partition: bool = True

    def instantiate(self, *args, **kwargs):
        return PartitionGridGaussianDensityControllerImpl(self)


class PartitionGridGaussianDensityControllerImpl(GridGaussianDensityControllerImpl):
    config: PartitionGridGaussianDensityController

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
            LoDGridDensityController.densify_anchors(self, grads_norm, primitive_mask, gaussian_model, optimizers)
        else:
            GridDensityController.densify_anchors_paperversion(
                self, grads_norm, primitive_mask, gaussian_model, optimizers
            )

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


class GridDensityController:
    @classmethod
    def densify_anchors(
        cls,
        controller: PartitionGridGaussianDensityControllerImpl,
        grads,
        primitive_mask,
        gaussian_model: GridGaussianModel,
        optimizers,
    ):
        grads[~primitive_mask] = 0.0  # only consider offsets that denom > denom_thresh
        grad_threshold = controller.config.densify_grad_threshold
        grad_mask = grads >= grad_threshold

        all_primitives = gaussian_model.get_anchors.unsqueeze(dim=1) + (
            gaussian_model.get_offsets * gaussian_model.get_scalings[:, :3].unsqueeze(dim=1)
        )

        filtered_res: CandidateAnchors = GridFilteringUtils.filter_primitives(
            gaussian_model=gaussian_model,
            all_primitives=all_primitives,
            grad_mask=grad_mask,
            overlap=controller.config.overlap,
        )

        if filtered_res.n_anchors > 0:
            partition_info = getattr(controller, "partition_info", None)
            property_dict = filtered_res.get_all_properties(
                gaussian_model,
                gaussian_model.voxel_size,
                scatter_mode=controller.config.scatter_mode,
                partition_info=partition_info if controller.config.densify_in_partition else None,
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

    @classmethod
    def densify_anchors_paperversion(
        cls,
        controller: PartitionGridGaussianDensityControllerImpl,
        grads,
        primitive_mask,
        gaussian_model: GridGaussianModel,
        optimizers,
    ):
        n_anchors_init, n_offsets = gaussian_model.get_anchors.shape[0], gaussian_model.config.n_offsets
        for i in range(gaussian_model.config.update_depth):
            cur_threshold = controller.config.densify_grad_threshold * (
                (gaussian_model.config.update_hierachy_factor // 2) ** i
            )
            candidate_mask = grads >= cur_threshold
            candidate_mask = torch.logical_and(candidate_mask, primitive_mask)

            rand_mask = torch.rand_like(candidate_mask.float()).to(candidate_mask.device) > (0.5 ** (i + 1))
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            n_anchors_diff = gaussian_model.get_anchors.shape[0] - n_anchors_init
            if n_anchors_diff <= 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat(
                    [candidate_mask, candidate_mask.new_zeros((n_anchors_diff * n_offsets,))], dim=0
                )

            all_primitives = gaussian_model.get_anchors.unsqueeze(dim=1) + (
                gaussian_model.get_offsets * gaussian_model.get_scalings[:, :3].unsqueeze(dim=1)
            )
            size_factor = gaussian_model.config.update_init_factor // (gaussian_model.config.update_hierachy_factor**i)
            cur_size = gaussian_model.voxel_size * size_factor

            filtered_res: CandidateAnchors = GridFilteringUtils.filter_primitives_paperversion(
                gaussian_model=gaussian_model,
                all_primitives=all_primitives,
                grad_mask=candidate_mask,
                voxel_size=cur_size,
                overlap=controller.config.overlap,
            )

            if filtered_res.n_anchors > 0:
                partition_info = getattr(controller, "partition_info", None)
                property_dict = filtered_res.get_all_properties(
                    gaussian_model,
                    cur_size,
                    scatter_mode=controller.config.scatter_mode,
                    partition_info=partition_info if controller.config.densify_in_partition else None,
                )
                new_properties = OptimizerManipulator.cat_tensors_to_properties(
                    property_dict, gaussian_model, optimizers
                )
                gaussian_model.properties = new_properties


class LoDGridDensityController:
    @classmethod
    def densify_anchors(
        cls,
        controller: PartitionGridGaussianDensityControllerImpl,
        grads,
        primitive_mask,
        gaussian_model: LoDGridGaussianModel,
        optimizers,
    ):
        n_anchors_init, n_offsets = gaussian_model.get_anchors.shape[0], gaussian_model.config.n_offsets
        grads[~primitive_mask] = 0.0  # only consider offsets that denom > denom_thresh
        anchor_grads = torch.sum(grads.reshape(-1, n_offsets), dim=-1) / (
            1e-6 + torch.sum(primitive_mask.reshape(-1, n_offsets), dim=-1)
        )
        for cur_level in range(gaussian_model._max_level):
            cls.densify_anchor_per_level(
                controller=controller,
                n_anchors_init=n_anchors_init,
                grads=grads,
                anchor_grads=anchor_grads,
                cur_level=cur_level,
                gaussian_model=gaussian_model,
                optimizers=optimizers,
            )

    @classmethod
    def densify_anchor_per_level(
        cls,
        controller: PartitionGridGaussianDensityControllerImpl,
        n_anchors_init: int,
        grads: torch.Tensor,
        anchor_grads: torch.Tensor,
        cur_level: int,
        gaussian_model: LoDGridGaussianModel,
        optimizers: List,
    ):
        n_offsets = gaussian_model.n_offsets

        update_value = gaussian_model.config.fork**controller.config.densification_ratio
        levels = gaussian_model.get_levels
        level_mask = levels == cur_level
        level_mask_ds = levels == cur_level + 1
        if torch.sum(level_mask) == 0:
            return
        cur_size = gaussian_model.voxel_size / (float(gaussian_model.config.fork) ** cur_level)
        ds_size = cur_size / gaussian_model.config.fork

        # update threshold
        cur_threshold = controller.config.densify_grad_threshold * (update_value**cur_level)
        ds_threshold = cur_threshold * update_value
        extra_threshold = cur_threshold * controller.config.extra_ratio
        # mask from grad threshold
        grad_mask = (grads >= cur_threshold) & (grads < ds_threshold)
        grad_mask_ds = grads >= ds_threshold
        grad_mask_extra = anchor_grads >= extra_threshold

        # if prev level add anchors, mask size will dismatch gaussian_model.get_anchors
        n_anchors_cur = gaussian_model.get_anchors.shape[0]
        n_anchors_diff = n_anchors_cur - n_anchors_init
        if n_anchors_diff > 0:
            grad_mask = torch.cat([grad_mask, grad_mask.new_zeros((n_anchors_diff * n_offsets,))], dim=0)
            grad_mask_ds = torch.cat([grad_mask_ds, grad_mask_ds.new_zeros((n_anchors_diff * n_offsets,))], dim=0)
            grad_mask_extra = torch.cat([grad_mask_extra, grad_mask_extra.new_zeros((n_anchors_diff,))], dim=0)

        # calculate grad mask: grad > thresh and level == current level (next level)
        level_mask_repeat = repeat(level_mask, "n -> (n o)", o=n_offsets)
        grad_mask = torch.logical_and(grad_mask, level_mask_repeat)
        grad_mask_ds = torch.logical_and(grad_mask_ds, level_mask_repeat)
        grad_mask_extra = torch.logical_and(grad_mask_extra, level_mask)

        # if all level are activated, decide whether to update extra_levels by anchor grad
        # in renderer, predicted levels is added by extra_levels
        # means that as the times anchor grad exceed thresh increases, the anchor will be considered more fined during rendering
        if gaussian_model.activate_level >= gaussian_model.max_level:
            gaussian_model.set_property(
                "extra_levels", gaussian_model.get_extra_levels + controller.config.extra_up * grad_mask_extra.float()
            )

        all_primitives = gaussian_model.get_anchors.unsqueeze(dim=1) + (
            gaussian_model.get_offsets * gaussian_model.get_scalings[:, :3].unsqueeze(dim=1)
        )
        filtered_res: CandidateAnchors = GridFilteringUtils.filter_lod_primitives(
            gaussian_model=gaussian_model,
            all_primitives=all_primitives,
            grad_mask=grad_mask,
            res_level=cur_level,
            level_mask=level_mask,
            cam_infos=controller.camera_infos,
            overlap=controller.config.overlap,
            is_next_level=False,
        )
        filtered_res_ds: CandidateAnchors = GridFilteringUtils.filter_lod_primitives(
            gaussian_model=gaussian_model,
            all_primitives=all_primitives,
            grad_mask=grad_mask_ds,
            res_level=cur_level + 1,
            level_mask=level_mask_ds,
            cam_infos=controller.camera_infos,
            overlap=controller.config.overlap,
            is_next_level=True,
        )

        # cat new anchors to gaussian_model and
        partition_info = getattr(controller, "partition_info", None)
        if filtered_res.n_anchors > 0:
            property_dict = filtered_res.get_all_properties(
                gaussian_model,
                cur_size,
                scatter_mode=controller.config.scatter_mode,
                partition_info=partition_info if controller.config.densify_in_partition else None,
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

        if filtered_res_ds.n_anchors > 0:
            property_dict = filtered_res_ds.get_all_properties(
                gaussian_model,
                ds_size,
                scatter_mode=controller.config.scatter_mode,
                partition_info=partition_info if controller.config.densify_in_partition else None,
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties


class GridFilteringUtils:
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

        return CandidateAnchors(
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

        return CandidateAnchors(
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

        return CandidateAnchors(
            anchors=filtered_anchors,
            levels=filtered_levels,
            grad_mask=grad_mask,
            unique_indices=unique_indices,
            keep_mask=keep_mask,
        )


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
        # anchor_features = repeat(gaussian_model.get_anchor_features, "n c -> (n o) c", o=gaussian_model.n_offsets)
        # # if anchors are added, shape of anchor_features may dismatch grad_mask
        # anchor_features = anchor_features[: len(self.grad_mask)][self.grad_mask]
        # # select max value of anchor features among primitives that convert to same grid
        # anchor_features = scatter_max(anchor_features, self.unique_indices.unsqueeze(1).expand(-1, anchor_features.shape[-1]), dim=0)[0]  # fmt: skip
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
        self,
        gaussian_model: GridGaussianModel,
        voxel_size: float,
        scatter_mode: Literal["max", "mean"] = "max",
        partition_info: PartitionInfo = None,
    ):
        property_dict = self.get_basic_properties(gaussian_model, voxel_size)
        if getattr(gaussian_model, "get_levels", None) is not None and gaussian_model.get_levels.shape[0] > 0:
            property_dict.update(self.get_lod_grid_properties(gaussian_model))
        if getattr(gaussian_model, "gaussian_mlps", None) is not None:
            property_dict.update(self.get_scaffold_properties(gaussian_model, scatter_mode))
        # TODO: explicit model

        if partition_info is not None:
            mask = partition_info.is_in_partition(self.anchors)
            _property_dict = {k: v[mask] for k, v in property_dict.items()}
            property_dict = _property_dict
        return property_dict
