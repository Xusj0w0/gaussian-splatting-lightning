from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
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

from .utils import GridFilteringUtils

__all__ = [
    "GridGaussianDensityController",
    "GridGaussianDensityControllerImpl",
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

    ssim_grad: bool = False

    def instantiate(self, *args, **kwargs):
        return GridGaussianDensityControllerImpl(self)


class GridGaussianDensityControllerImpl(VanillaDensityControllerImpl):
    GRID_FILTERING_UTILS = GridFilteringUtils

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

            bs = pl_module.trainer.datamodule.hparams.get("batch_size", 1)
            mv = pl_module.trainer.datamodule.hparams.get("multiview", False)
            # self.batch_size = bs * (2 if mv else 1)
            self.batch_size = bs

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
        if self.config.ssim_grad:
            metric = getattr(pl_module, "_current_metrics", None)
            if metric is not None:
                grad = torch.autograd.grad(1.0 - metric["ssim"], outputs["viewspace_points"], retain_graph=True)[0]
                # scale = metric["loss"].item() / (1.0 - metric["ssim"].item() + 1e-8)
                # scale = np.clip(scale, 0.5, 2.0)
                # self._means2d_grad_ssim = grad * scale
                self._means2d_grad_ssim = grad
            else:
                self._means2d_grad_ssim = None

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
        anchor_indices = (primitive_indices.float() / n_offsets).long().clamp(0, n_anchors - 1)
        self._anchor_opacity_accum += scatter_sum(opacities, anchor_indices, dim=0, dim_size=n_anchors)
        self._anchor_denom += (
            torch.bincount(anchor_mask.clamp(0, n_anchors - 1), minlength=n_anchors).float() / self.batch_size
        )

        if self.config.absgrad:
            xys_grad = viewspace_point_tensor.absgrad
        elif self.config.ssim_grad and self._means2d_grad_ssim is not None:
            scale = (
                torch.norm(viewspace_point_tensor.grad, dim=-1).mean()
                / (torch.norm(self._means2d_grad_ssim, dim=-1) + 1e-8).mean()
            ).item()
            scale = np.clip(scale, 0.5, 2.0)
            xys_grad = self._means2d_grad_ssim * scale
            # top2_scales = torch.topk(outputs["scales"], k=2, dim=1).values
            # axis_ratio_filter = top2_scales[..., 0] < 10 * top2_scales[..., 1]
            # xys_grad = xys_grad[axis_ratio_filter[visibility_filter]]
            # visibility_filter = visibility_filter & axis_ratio_filter
            # xys_grad[~axis_ratio_filter[visibility_filter]] = 0.0
        else:
            xys_grad = viewspace_point_tensor.grad
        xys_grad = xys_grad[..., :2]
        if viewspace_points_grad_scale is not None:
            xys_grad = xys_grad * viewspace_points_grad_scale
        grad_norm = torch.norm(xys_grad, dim=-1)
        proj_indices = primitive_indices[visibility_filter].clamp(0, n_anchors * n_offsets - 1)
        self._primitive_gradient_accum += scatter_sum(grad_norm, proj_indices, dim=0, dim_size=n_anchors * n_offsets)
        self._primitive_denom += torch.bincount(proj_indices, minlength=n_anchors * n_offsets).float() / self.batch_size

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

    def after_density_changed(self, gaussian_model, optimizers, pl_module):
        self.register_density_states()

    def density_anchors(self, grads, primitive_mask, gaussian_model: GridGaussianModel, optimizers):
        grads[~primitive_mask] = 0.0  # only consider offsets that denom > denom_thresh
        grad_threshold = self.config.densify_grad_threshold
        grad_mask = grads >= grad_threshold

        all_primitives = gaussian_model.get_anchors.unsqueeze(dim=1) + (
            gaussian_model.get_offsets * gaussian_model.get_scalings[:, :3].unsqueeze(dim=1)
        )

        filtered_res = self.GRID_FILTERING_UTILS.filter_primitives(
            gaussian_model=gaussian_model,
            all_primitives=all_primitives,
            grad_mask=grad_mask,
            overlap=self.config.overlap,
        )

        if filtered_res.n_anchors > 0:
            property_dict = filtered_res.get_all_properties(
                gaussian_model, gaussian_model.voxel_size, scatter_mode=self.config.scatter_mode
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

    def densify_anchors_paperversion(
        self,
        grads,
        primitive_mask,
        gaussian_model: GridGaussianModel,
        optimizers,
    ):
        n_anchors_init, n_offsets = gaussian_model.get_anchors.shape[0], gaussian_model.config.n_offsets
        for i in range(gaussian_model.config.update_depth):
            cur_threshold = self.config.densify_grad_threshold * (
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

            filtered_res = self.GRID_FILTERING_UTILS.filter_primitives_paperversion(
                gaussian_model=gaussian_model,
                all_primitives=all_primitives,
                grad_mask=candidate_mask,
                voxel_size=cur_size,
                overlap=self.config.overlap,
            )

            if filtered_res.n_anchors > 0:
                property_dict = filtered_res.get_all_properties(
                    gaussian_model, cur_size, scatter_mode=self.config.scatter_mode
                )
                new_properties = OptimizerManipulator.cat_tensors_to_properties(
                    property_dict, gaussian_model, optimizers
                )
                gaussian_model.properties = new_properties

    def densify_anchors_multilevel(
        self,
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
            self.densify_anchor_per_level(
                n_anchors_init=n_anchors_init,
                grads=grads,
                anchor_grads=anchor_grads,
                cur_level=cur_level,
                gaussian_model=gaussian_model,
                optimizers=optimizers,
            )

    def densify_anchor_per_level(
        self,
        n_anchors_init: int,
        grads: torch.Tensor,
        anchor_grads: torch.Tensor,
        cur_level: int,
        gaussian_model: LoDGridGaussianModel,
        optimizers: List,
    ):
        n_offsets = gaussian_model.n_offsets

        update_value = gaussian_model.config.fork**self.config.densification_ratio
        levels = gaussian_model.get_levels
        level_mask = levels == cur_level
        level_mask_ds = levels == cur_level + 1
        if torch.sum(level_mask) == 0:
            return
        cur_size = gaussian_model.voxel_size / (float(gaussian_model.config.fork) ** cur_level)
        ds_size = cur_size / gaussian_model.config.fork

        # update threshold
        cur_threshold = self.config.densify_grad_threshold * (update_value**cur_level)
        ds_threshold = cur_threshold * update_value
        extra_threshold = cur_threshold * self.config.extra_ratio
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
                "extra_levels", gaussian_model.get_extra_levels + self.config.extra_up * grad_mask_extra.float()
            )

        all_primitives = gaussian_model.get_anchors.unsqueeze(dim=1) + (
            gaussian_model.get_offsets * gaussian_model.get_scalings[:, :3].unsqueeze(dim=1)
        )
        filtered_res = self.GRID_FILTERING_UTILS.filter_lod_primitives(
            gaussian_model=gaussian_model,
            all_primitives=all_primitives,
            grad_mask=grad_mask,
            res_level=cur_level,
            level_mask=level_mask,
            cam_infos=self.camera_infos,
            overlap=self.config.overlap,
            is_next_level=False,
        )
        filtered_res_ds = self.GRID_FILTERING_UTILS.filter_lod_primitives(
            gaussian_model=gaussian_model,
            all_primitives=all_primitives,
            grad_mask=grad_mask_ds,
            res_level=cur_level + 1,
            level_mask=level_mask_ds,
            cam_infos=self.camera_infos,
            overlap=self.config.overlap,
            is_next_level=True,
        )

        # cat new anchors to gaussian_model and
        if filtered_res.n_anchors > 0:
            property_dict = filtered_res.get_all_properties(
                gaussian_model, cur_size, scatter_mode=self.config.scatter_mode
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

        if filtered_res_ds.n_anchors > 0:
            property_dict = filtered_res_ds.get_all_properties(
                gaussian_model, ds_size, scatter_mode=self.config.scatter_mode
            )
            new_properties = OptimizerManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

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

    @property
    def camera_infos(self) -> torch.Tensor:
        self._camera_infos: torch.Tensor
        return self._camera_infos
