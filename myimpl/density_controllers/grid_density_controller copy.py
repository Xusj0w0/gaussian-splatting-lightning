from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from einops import repeat
from lightning import LightningModule
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from internal.cameras.cameras import Cameras
from internal.density_controllers.density_controller import \
    Utils as OptimizerManipulator
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from myimpl.models.grid_gaussians import (GridFactory, GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)

__all__ = [
    "GridGaussianDensityController",
    "GridGaussianDensityControllerImpl",
    "GridFilteringUtils",
    "CandidateAnchors",
]


@dataclass
class GridGaussianDensityController(VanillaDensityController):
    densification: bool = True

    overlap: int = 1
    """maximum number of overlap anchors, <0 for no limit"""

    success_threshold: float = 0.8

    densification_ratio: float = 0.2

    extra_ratio: float = 0.25

    extra_up: float = 0.02

    update_from_iter: int = 500

    densify_from_iter: int = 1_500

    densify_until_iter: int = 30_000

    scatter_mode: Literal["max", "mean"] = "max"

    def instantiate(self, *args, **kwargs):
        return GridGaussianDensityControllerImpl(self)


class GridGaussianDensityControllerImpl(VanillaDensityControllerImpl):
    config: GridGaussianDensityController

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        density_states = ["_primitive_gradient_accum", "_primitive_denom", "_anchor_opacity_accum", "_anchor_denom"]
        self._density_state_names = tuple(density_states)

    @property
    def density_state_names(self):
        return self._density_state_names

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        if stage == "fit":
            device = pl_module.device
            n_anchors = pl_module.gaussian_model.get_anchors.shape[0]
            n_offsets = pl_module.gaussian_model.config.n_offsets
            self._init_density_state(n_anchors, n_offsets, device)

            cameras: Cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
            cam_centers = cameras.camera_center
            camera_infos = torch.cat([cam_centers, cam_centers.new_ones((cam_centers.shape[0], 1))], dim=-1)
            self.register_buffer("_camera_infos", camera_infos)

            self.batch_size = pl_module.trainer.datamodule.hparams["batch_size"]

    def _init_state(self, n_gaussians: int, device):
        pass

    def before_backward(
        self,
        outputs: dict,
        batch,
        gaussian_model: GridGaussianModel,
        optimizers: List,
        global_step: int,
        pl_module: LightningModule,
    ) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        outputs["viewspace_points"].retain_grad()

    def after_backward(
        self,
        outputs: dict,
        batch,
        gaussian_model: GridGaussianModel,
        optimizers: List,
        global_step: int,
        pl_module: LightningModule,
    ) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        if global_step >= self.config.update_from_iter:
            with torch.no_grad():
                self.update_state(outputs, gaussian_model.n_anchors, gaussian_model.n_offsets)

                if (
                    self.config.densification
                    and global_step >= self.config.densify_from_iter
                    and global_step % self.config.densification_interval == 0
                ):
                    # filter out mlp optimizers
                    property_optimizers = []
                    for opt in optimizers:
                        if not any(["mlp" in pg["name"] for pg in opt.param_groups]):
                            property_optimizers.append(opt)

                    self._densify_and_prune(gaussian_model=gaussian_model, optimizers=property_optimizers)

    def _init_density_state(self, n_anchors: int, n_offsets: int, device):
        self._primitive_gradient_accum: torch.Tensor; self._primitive_denom: torch.Tensor  # fmt: skip
        self._anchor_opacity_accum: torch.Tensor; self._anchor_denom: torch.Tensor  # fmt: skip

        self.register_buffer("_anchor_opacity_accum", torch.zeros((n_anchors,), device=device))
        self.register_buffer("_anchor_denom", torch.zeros((n_anchors,), device=device))
        self.register_buffer("_primitive_gradient_accum", torch.zeros((n_anchors * n_offsets,), device=device))
        self.register_buffer("_primitive_denom", torch.zeros((n_anchors * n_offsets,), device=device))

    def register_density_states(self):
        for name in self.density_state_names:
            self.register_buffer(name, getattr(self, name))

    def update_state(self, outputs, n_anchors, n_offsets):
        viewspace_point_tensor = outputs["viewspace_points"]
        anchor_mask, primitive_mask, visibility_filter = (
            outputs["anchor_mask"],  # [N, ], mask anchor by pred level and pro-projection
            outputs["primitive_mask"],  # [N'*n_offsets, ], mask primitives by opacities(>0), N'=anchor_mask.sum()
            outputs["visibility_filter"],  # [M, ], mask primitives further by projection, M=primitive_mask.sum()
        )
        opacities: torch.Tensor = outputs["opacities"]
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        indices = torch.arange(n_anchors * n_offsets, device=opacities.device)
        indices = indices.reshape(-1, n_offsets)[anchor_mask].reshape(-1)
        primitive_indices = indices[primitive_mask]
        anchor_indices = (primitive_indices.float() / n_offsets).long()
        self._anchor_opacity_accum += scatter_sum(opacities, anchor_indices, dim=0, dim_size=n_anchors)
        self._anchor_denom += torch.bincount(anchor_indices, minlength=n_anchors).float() / self.batch_size

        # indices = torch.arange(n_anchors, device=opacities.device)
        # indices = indices[anchor_mask].unsqueeze(dim=-1).repeat(1, n_offsets).view(-1)
        # anchor_indices = indices[primitive_mask]
        # anchor_opacities = scatter_sum(opacities, anchor_indices, dim=0, dim_size=n_anchors)
        # anchor_denom = torch.bincount(indices, minlength=n_anchors).float() / self.batch_size
        # self._anchor_opacity_accum += anchor_opacities
        # self._anchor_denom += anchor_denom

        # _opacities = opacities.new_zeros((primitive_mask.shape[0]))
        # _opacities[primitive_mask] = opacities.clone().view(-1).detach()
        # _opacities = _opacities.view(-1, n_offsets)
        # self._anchor_opacity_accum[anchor_mask] += _opacities.sum(dim=-1, keepdim=True)
        # self._anchor_denom[anchor_mask] += 1

        # anchor_mask_repeat = torch.repeat_interleave(anchor_mask, n_offsets, dim=0)
        # _anchor_mask = primitive_mask.new_zeros((n_anchors,), dtype=torch.bool)
        # _anchor_mask[anchor_mask] = True
        # anchor_mask_repeat = repeat(_anchor_mask, "n -> (n o)", o=n_offsets)
        # combined_mask = primitive_mask.new_zeros((self._primitive_gradient_accum.shape[0],))
        # combined_mask[anchor_mask_repeat] = primitive_mask
        # combined_mask[combined_mask.clone()] = visibility_filter

        # TODO: use packed=True, xys_grad is filtered in projection process
        xys_grad = viewspace_point_tensor.absgrad if self.config.absgrad else viewspace_point_tensor.grad
        xys_grad = xys_grad[..., :2]
        if viewspace_points_grad_scale is not None:
            xys_grad = xys_grad * viewspace_points_grad_scale
        grad_norm = torch.norm(xys_grad, dim=-1)
        # self._primitive_gradient_accum[combined_mask] += grad_norm
        # self._primitive_denom[combined_mask] += 1
        projection_indices = primitive_indices[visibility_filter]
        self._primitive_gradient_accum += scatter_sum(
            grad_norm, projection_indices, dim=0, dim_size=n_anchors * n_offsets
        )
        self._primitive_denom += (
            torch.bincount(projection_indices, minlength=n_anchors * n_offsets).float() / self.batch_size
        )

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
        opacity_mask = self._anchor_opacity_accum < self.config.cull_opacity_threshold
        denom_mask = self._anchor_denom > anchor_denom_thresh
        keep_mask = ~torch.logical_and(denom_mask, opacity_mask)

        # re-accumulate: set denom and opacity_accum of anchors that denom > denom_thresh to 0
        # prune anchors that denom > denom_thresh and opacity_accum < opacity_thresh
        self._prune_anchors(keep_mask, gaussian_model, optimizers)

        # prune buffers
        self._prune_buffers(denom_mask, keep_mask, n_offsets)
        torch.cuda.empty_cache()

        # register buffer
        self.register_density_states()

    def _prune_anchors(self, keep_mask, gaussian_model: GridGaussianModel, optimizers):
        new_properties = OptimizerManipulator.prune_properties(keep_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_properties

    def _densify_buffers(self, num_anchors, num_primitives, primitive_mask):
        self._anchor_denom = torch.cat([self._anchor_denom, self._anchor_denom.new_zeros((num_anchors,))], dim=0)
        self._anchor_opacity_accum = torch.cat(
            [self._anchor_opacity_accum, self._anchor_opacity_accum.new_zeros((num_anchors,))], dim=0
        )
        self._primitive_denom[primitive_mask] = 0
        self._primitive_gradient_accum[primitive_mask] = 0.0
        self._primitive_denom = torch.cat(
            [self._primitive_denom, self._primitive_denom.new_zeros((num_primitives,))], dim=0
        )
        self._primitive_gradient_accum = torch.cat(
            [self._primitive_gradient_accum, self._primitive_gradient_accum.new_zeros((num_primitives,))], dim=0
        )

    def _prune_buffers(self, denom_mask, keep_mask, n_offsets):
        self._primitive_denom = self._primitive_denom.view(-1, n_offsets)[keep_mask].view(-1)
        self._primitive_gradient_accum = self._primitive_gradient_accum.view(-1, n_offsets)[keep_mask].view(-1)
        if denom_mask.sum() > 0:
            self._anchor_denom[denom_mask] = 0
            self._anchor_opacity_accum[denom_mask] = 0.0
        self._anchor_denom = self._anchor_denom[keep_mask]
        self._anchor_opacity_accum = self._anchor_opacity_accum[keep_mask]

    def on_load_checkpoint(self, module, checkpoint):
        density_state_dict = {
            k.replace("density_controller.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("density_controller")
        }
        assert (
            "_anchor_denom" in density_state_dict and "_primitive_denom" in density_state_dict
        ), "Density controller states not found in checkpoint"
        n_anchors = density_state_dict["_anchor_denom"].shape[0]
        n_offsets = density_state_dict["_primitive_denom"].shape[0] // n_anchors
        self._init_density_state(n_anchors, n_offsets, device="cpu")

        assert "_camera_infos" in density_state_dict, "Camera infos not found in checkpoint"
        self.register_buffer("_camera_infos", torch.zeros_like(density_state_dict["_camera_infos"]))

        super().load_state_dict(density_state_dict)

    def after_density_changed(self, gaussian_model, optimizers, pl_module):
        self.register_density_states()

    @property
    def camera_infos(self) -> torch.Tensor:
        self._camera_infos: torch.Tensor
        return self._camera_infos


class GridDensityController:
    @classmethod
    def densify_anchors(
        cls,
        controller: GridGaussianDensityControllerImpl,
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
            property_dict = filtered_res.get_all_properties(
                gaussian_model, gaussian_model.voxel_size, scatter_mode=controller.config.scatter_mode
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

    @classmethod
    def densify_anchors_paperversion(
        cls,
        controller: GridGaussianDensityControllerImpl,
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
                property_dict = filtered_res.get_all_properties(
                    gaussian_model, cur_size, scatter_mode=controller.config.scatter_mode
                )
                new_properties = OptimizerManipulator.cat_tensors_to_properties(
                    property_dict, gaussian_model, optimizers
                )
                gaussian_model.properties = new_properties


class LoDGridDensityController:
    @classmethod
    def densify_anchors(
        cls,
        controller: GridGaussianDensityControllerImpl,
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
        controller: GridGaussianDensityControllerImpl,
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
        if filtered_res.n_anchors > 0:
            property_dict = filtered_res.get_all_properties(
                gaussian_model, cur_size, scatter_mode=controller.config.scatter_mode
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

        if filtered_res_ds.n_anchors > 0:
            property_dict = filtered_res_ds.get_all_properties(
                gaussian_model, ds_size, scatter_mode=controller.config.scatter_mode
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
        anchor_features = repeat(gaussian_model.get_anchor_features, "n c -> (n o) c", o=gaussian_model.n_offsets)
        # if anchors are added, shape of anchor_features may dismatch grad_mask
        anchor_features = anchor_features[: len(self.grad_mask)][self.grad_mask]

        # select value of anchor features among primitives that convert to same grid
        if scatter_mode == "max":
            anchor_features = scatter_max(anchor_features, self.unique_indices.unsqueeze(1).expand(-1, anchor_features.shape[-1]), dim=0)[0]  # fmt: skip
        elif scatter_mode == "mean":
            anchor_features = scatter_mean(anchor_features, self.unique_indices.unsqueeze(1).expand(-1, anchor_features.shape[-1]), dim=0)[0]  # fmt: skip
        else:
            raise ValueError(f"scatter_mode {scatter_mode} not supported")
        anchor_features = anchor_features[self.keep_mask]

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
