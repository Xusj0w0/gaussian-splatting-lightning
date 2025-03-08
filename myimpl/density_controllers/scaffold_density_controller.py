from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from einops import repeat
from lightning import LightningModule
from torch import nn
from torch_scatter import scatter_max

from internal.density_controllers.density_controller import \
    Utils as DensityUtils
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from myimpl.models.scaffold_gaussian import ScaffoldLoDGaussianModel


@dataclass
class ScaffoldDensityController(VanillaDensityController):
    densification: bool = True

    overlap: bool = False

    success_threshold: float = 0.8

    densification_ratio: float = 0.2

    extra_ratio: float = 0.25

    extra_up: float = 0.02

    update_from_iter: int = 500

    densify_from_iter: int = 1_500

    densify_until_iter: int = 25_000

    def instantiate(self, *args, **kwargs):
        return ScaffoldDensityControllerImpl(self)


class ScaffoldDensityControllerImpl(VanillaDensityControllerImpl):
    config: ScaffoldDensityController

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        buffer_names = ["_offset_gradient_accum", "_offset_denom", "_opacity_accum", "_denom"]
        self._buffer_names = tuple(buffer_names)

    @property
    def buffer_names(self):
        return self._buffer_names

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        device = pl_module.device
        n_anchors = pl_module.gaussian_model.get_anchors.shape[0]
        n_offsets = pl_module.gaussian_model.config.n_offsets

        self._offset_gradient_accum: torch.Tensor; self._offset_denom: torch.Tensor  # fmt: skip
        self._opacity_accum: torch.Tensor; self._denom: torch.Tensor  # fmt: skip

        self.register_buffer("_offset_gradient_accum", torch.zeros((n_anchors * n_offsets, 1), device=device))
        self.register_buffer("_offset_denom", torch.zeros((n_anchors * n_offsets, 1), device=device))
        self.register_buffer("_opacity_accum", torch.zeros((n_anchors, 1), device=device))
        self.register_buffer("_denom", torch.zeros((n_anchors, 1), device=device))

    def before_backward(
        self,
        outputs: dict,
        batch,
        gaussian_model: ScaffoldLoDGaussianModel,
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
        gaussian_model: ScaffoldLoDGaussianModel,
        optimizers: List,
        global_step: int,
        pl_module: LightningModule,
    ) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        if global_step >= self.config.update_from_iter:
            with torch.no_grad():
                self.update_state(outputs)

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

    def _init_state(self, n_gaussians: int, device):
        pass

    def register_accums_and_denoms(self):
        for name in self.buffer_names:
            self.register_buffer(name, getattr(self, name))

    def update_state(self, outputs):
        viewspace_point_tensor = outputs["viewspace_points"]
        anchor_mask, offset_mask, visibility_filter = (
            outputs["anchor_mask"],  # [N, ], mask anchor by pred level and pro-projection
            outputs["offset_mask"],  # [N'*n_offsets, ], mask primitives by opacities(>0), N'=anchor_mask.sum()
            outputs["visibility_filter"],  # [M, ], mask primitives further by projection, M=offset_mask.sum()
        )
        opacities: torch.Tensor = outputs["opacities"]
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        n_anchors = anchor_mask.shape[0]
        n_offsets = int(offset_mask.shape[0] / anchor_mask.sum().item())

        # debug
        _opacities = opacities.new_zeros((offset_mask.shape[0]))
        _opacities[offset_mask] = opacities.clone().view(-1).detach()
        _opacities = _opacities.view(-1, n_offsets)
        self._opacity_accum[anchor_mask] += _opacities.sum(dim=-1, keepdim=True)
        self._denom[anchor_mask] += 1

        # anchor_mask_repeat = torch.repeat_interleave(anchor_mask, n_offsets, dim=0)
        anchor_mask_repeat = repeat(anchor_mask, "n -> (n o)", o=n_offsets)
        combined_mask = offset_mask.new_zeros((self._offset_gradient_accum.shape[0],))
        combined_mask[anchor_mask_repeat] = offset_mask
        combined_mask[combined_mask.clone()] = visibility_filter
        xys_grad = viewspace_point_tensor.absgrad if self.config.absgrad else viewspace_point_tensor.grad
        xys_grad = xys_grad[visibility_filter, :2]
        if viewspace_points_grad_scale is not None:
            xys_grad = xys_grad * viewspace_points_grad_scale
        grad_norm = torch.norm(xys_grad, dim=-1, keepdim=True)
        self._offset_gradient_accum[combined_mask] += grad_norm
        self._offset_denom[combined_mask] += 1

    def _densify_and_prune(self, gaussian_model: ScaffoldLoDGaussianModel, optimizers: List):
        grads = self._offset_gradient_accum / self._offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        denom_thresh = self.config.densification_interval * self.config.success_threshold * 0.5
        offset_mask = (self._offset_denom > denom_thresh).squeeze(dim=1)

        # densify anchors
        self._densify_anchors(
            grads=grads_norm,
            offset_mask=offset_mask,
            gaussian_model=gaussian_model,
            optimizers=optimizers,
        )

        # prune anchors
        anchor_denom_thresh = self.config.densification_interval * self.config.success_threshold
        opacity_mask = (self._opacity_accum < self.config.cull_opacity_threshold).squeeze(1)
        denom_mask = (self._denom > anchor_denom_thresh).squeeze(1)

        # re-accumulate: set denom and opacity_accum of anchors that denom > denom_thresh to 0
        # prune anchors that denom > denom_thresh and opacity_accum < opacity_thresh
        self._prune_anchors(denom_mask, opacity_mask, gaussian_model, optimizers)

        # register buffer
        self.register_accums_and_denoms()

    def _densify_anchor_single_level(
        self,
        n_anchors_init: int,
        grads: torch.Tensor,
        anchor_grads: torch.Tensor,
        cur_level: int,
        gaussian_model: ScaffoldLoDGaussianModel,
    ):
        n_offsets = gaussian_model.n_offsets
        reach_max_level = gaussian_model.activate_level >= gaussian_model.max_level

        update_value = gaussian_model.config.fork**self.config.densification_ratio
        levels = gaussian_model.get_levels
        level_mask = levels == cur_level
        level_ds_mask = levels == cur_level + 1
        if torch.sum(level_mask) == 0:
            return
        cur_size = gaussian_model.voxel_size / (float(gaussian_model.config.fork) ** cur_level)
        ds_size = cur_size / gaussian_model.config.fork

        # update threshold
        cur_threshold = self.config.densify_grad_threshold * (update_value**cur_level)
        ds_threshold = cur_threshold * update_value
        extra_threshold = cur_threshold * self.config.extra_ratio
        # mask from grad threshold
        candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
        candidate_ds_mask = grads >= ds_threshold
        candidate_extra_mask = anchor_grads >= extra_threshold

        n_anchors_cur = gaussian_model.get_anchors.shape[0]
        n_anchors_diff = n_anchors_cur - n_anchors_init
        if n_anchors_diff > 0:
            candidate_mask = torch.cat([candidate_mask, candidate_mask.new_zeros((n_anchors_diff * n_offsets,))], dim=0)
            candidate_ds_mask = torch.cat(
                [candidate_ds_mask, candidate_ds_mask.new_zeros((n_anchors_diff * n_offsets,))], dim=0
            )
            candidate_extra_mask = torch.cat(
                [candidate_extra_mask, candidate_extra_mask.new_zeros((n_anchors_diff,))], dim=0
            )

        # level_mask_repeat = torch.repeat_interleave(level_mask, n_offsets, dim=0)
        level_mask_repeat = repeat(level_mask, "n -> (n o)", o=n_offsets)
        candidate_mask = torch.logical_and(candidate_mask, level_mask_repeat)
        candidate_ds_mask = torch.logical_and(candidate_ds_mask, level_mask_repeat)
        candidate_extra_mask = torch.logical_and(candidate_extra_mask, level_mask)
        if not gaussian_model.config.optimization.progressive or reach_max_level:
            gaussian_model.set_property(
                "extra_levels", gaussian_model.get_extra_levels + self.config.extra_up * candidate_extra_mask.float()
            )

        all_xyz = gaussian_model.get_anchors.unsqueeze(dim=1) + (
            gaussian_model.get_offsets * gaussian_model.get_scalings[:, :3].unsqueeze(dim=1)
        )

        # anchors grid of current voxel_size
        grid_coords = gaussian_model.xyz2grid(gaussian_model.get_anchors[level_mask], cur_size)
        # satisfied offsets positions grid
        selected_xyz = all_xyz.view(-1, 3)[candidate_mask]
        selected_grid_coords = gaussian_model.xyz2grid(selected_xyz, cur_size)
        selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
        candidate_anchor, new_levels, remove_duplicates = self._get_valid_grids(
            grid_coords, selected_grid_coords_unique, gaussian_model, cur_size, cur_level
        )

        grid_coords_ds = gaussian_model.xyz2grid(gaussian_model.get_anchors[level_ds_mask], ds_size)
        selected_xyz_ds = all_xyz.view(-1, 3)[candidate_ds_mask]
        selected_grid_coords_ds = gaussian_model.xyz2grid(selected_xyz_ds, ds_size)
        selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(
            selected_grid_coords_ds, return_inverse=True, dim=0
        )
        if (
            not gaussian_model.config.optimization.progressive or reach_max_level
        ) and cur_level < gaussian_model._max_level - 1:
            candidate_anchor_ds, new_levels_ds, remove_duplicates_ds = self._get_valid_grids(
                grid_coords_ds, selected_grid_coords_unique_ds, gaussian_model, ds_size, cur_level + 1
            )
        else:
            remove_duplicates_ds = level_ds_mask.new_zeros((selected_grid_coords_unique_ds.shape[0],))
            candidate_anchor_ds = selected_grid_coords_unique_ds.new_zeros((0, 3))
            new_levels_ds = levels.new_zeros((0,))

        if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
            num_c, num_c_ds = candidate_anchor.shape[0], candidate_anchor_ds.shape[0]

            new_anchors = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
            new_levels = torch.cat([new_levels, new_levels_ds], dim=0)

            new_feat = (
                gaussian_model.get_anchor_features.unsqueeze(1)
                .repeat(1, n_offsets, 1)
                .view(-1, gaussian_model.config.feature_dim)[candidate_mask]
            )
            new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]  # fmt: skip
            new_feat_ds = (
                gaussian_model.get_anchor_features.unsqueeze(1)
                .repeat(1, n_offsets, 1)
                .view(-1, gaussian_model.config.feature_dim)[candidate_ds_mask]
            )
            new_feat_ds = scatter_max(new_feat_ds, inverse_indices_ds.unsqueeze(1).expand(-1, new_feat_ds.size(1)), dim=0)[0][remove_duplicates_ds]  # fmt: skip
            new_feat = torch.cat([new_feat, new_feat_ds], dim=0)

            new_scales = gaussian_model.get_scalings.new_ones((num_c, 6)) * cur_size
            new_scales_ds = gaussian_model.get_scalings.new_ones((num_c_ds, 6)) * ds_size
            new_scales = torch.cat([new_scales, new_scales_ds], dim=0)
            new_scales = gaussian_model.scale_inverse_activation(new_scales)

            new_rotations = gaussian_model.get_rotations.new_zeros((num_c, 4))
            new_rotations_ds = gaussian_model.get_rotations.new_zeros((num_c_ds, 4))
            new_rotations = torch.cat([new_rotations, new_rotations_ds], dim=0)
            new_rotations[:, 0] = 1.0

            new_offsets = gaussian_model.get_offsets.new_zeros((num_c, 1, 3)).repeat(1, n_offsets, 1)
            new_offsets_ds = gaussian_model.get_offsets.new_zeros((num_c_ds, 1, 3)).repeat(1, n_offsets, 1)
            new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

            new_extra_levels = gaussian_model.get_extra_levels.new_zeros((num_c,))
            new_extra_levels_ds = gaussian_model.get_extra_levels.new_zeros((num_c_ds,))
            new_extra_levels = torch.cat([new_extra_levels, new_extra_levels_ds], dim=0)

            new_properties = {
                "means": new_anchors,
                "scales": new_scales,
                "offsets": new_offsets,
                "levels": new_levels,
                "extra_levels": new_extra_levels,
                "rotations": new_rotations,
                "anchor_features": new_feat,
            }
            return new_anchors.shape[0], new_properties

    def _get_valid_grids(
        self,
        orig_grid: torch.Tensor,
        candidate_grid: torch.Tensor,
        gaussian_model: ScaffoldLoDGaussianModel,
        voxel_size: torch.Tensor,
        res_level: torch.Tensor,
    ):
        if self.config.overlap:
            remove_duplicates = orig_grid.new_ones((candidate_grid.shape[0],), dtype=torch.bool)
            candidate_anchor = gaussian_model.grid2xyz(candidate_grid, voxel_size)
            new_levels = gaussian_model.get_levels.new_ones((candidate_anchor.shape[0],)) * res_level
            weed_mask = gaussian_model.weed_out(candidate_anchor, new_levels, gaussian_model.visibility_threshold)
            candidate_anchor, new_levels = candidate_anchor[weed_mask], new_levels[weed_mask]
            remove_duplicates[remove_duplicates.clone()] = weed_mask
        elif candidate_grid.shape[0] > 0 and orig_grid.shape[0] > 0:
            remove_duplicates = ~self.get_remove_duplicates(orig_grid, candidate_grid)
            candidate_anchor = gaussian_model.grid2xyz(candidate_grid[remove_duplicates], voxel_size)
            new_levels = gaussian_model.get_levels.new_ones((candidate_anchor.shape[0],)) * res_level
            weed_mask = gaussian_model.weed_out(candidate_anchor, new_levels, gaussian_model.visibility_threshold)
            candidate_anchor, new_levels = candidate_anchor[weed_mask], new_levels[weed_mask]
            remove_duplicates[remove_duplicates.clone()] = weed_mask
        else:
            remove_duplicates = orig_grid.new_zeros((candidate_grid.shape[0],), dtype=torch.bool)
            candidate_anchor = candidate_grid.new_zeros((0, 3))
            new_levels = gaussian_model.get_levels.new_zeros((0,))
        return candidate_anchor, new_levels, remove_duplicates

    def _densify_anchors(self, grads, offset_mask, gaussian_model: ScaffoldLoDGaussianModel, optimizers):
        n_anchors_init, n_offsets = gaussian_model.get_anchors.shape[0], gaussian_model.config.n_offsets
        grads[~offset_mask] = 0.0  # only consider offsets that denom > denom_thresh
        anchor_grads = torch.sum(grads.reshape(-1, n_offsets), dim=-1) / (
            1e-6 + torch.sum(offset_mask.reshape(-1, n_offsets), dim=-1)
        )
        for cur_level in range(gaussian_model._max_level):
            output_pkg = self._densify_anchor_single_level(
                n_anchors_init=n_anchors_init,
                grads=grads,
                anchor_grads=anchor_grads,
                cur_level=cur_level,
                gaussian_model=gaussian_model,
            )
            if output_pkg is None:
                continue

            num_new_anchors, new_properties = output_pkg
            new_parameters = DensityUtils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
            gaussian_model.properties = new_parameters

            # update anchor level opacity accum and denom
            self._denom = torch.cat([self._denom, self._denom.new_zeros((num_new_anchors, 1))], dim=0)
            self._opacity_accum = torch.cat([self._opacity_accum, self._opacity_accum.new_zeros((num_new_anchors, 1))], dim=0)  # fmt: skip

            torch.cuda.empty_cache()

        # update xyz grad accum and denom
        self._offset_denom[offset_mask] = 0
        padding_denom = self._offset_denom.new_zeros(
            (gaussian_model.get_anchors.shape[0] * gaussian_model.n_offsets - self._offset_denom.shape[0], 1),
        )
        self._offset_denom = torch.cat([self._offset_denom, padding_denom], dim=0)

        self._offset_gradient_accum[offset_mask] = 0.0
        padding_accum = self._offset_gradient_accum.new_zeros(
            (gaussian_model.get_anchors.shape[0] * gaussian_model.n_offsets - self._offset_gradient_accum.shape[0], 1)
        )
        self._offset_gradient_accum = torch.cat([self._offset_gradient_accum, padding_accum], dim=0)

    def _prune_anchors(self, denom_mask, opacity_mask, gaussian_model: ScaffoldLoDGaussianModel, optimizers):
        n_offsets = gaussian_model.n_offsets

        keep = ~torch.logical_and(denom_mask, opacity_mask)
        new_parameters = DensityUtils.prune_properties(keep, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

        # update accum and denom
        # offset_gradient and denom
        self._offset_denom = self._offset_denom.view(-1, n_offsets)[keep].view(-1, 1)
        self._offset_gradient_accum = self._offset_gradient_accum.view(-1, n_offsets)[keep].view(-1, 1)
        # re-accumulate denom > denom_thresh
        if denom_mask.sum() > 0:
            self._denom[denom_mask] = 0
            self._opacity_accum[denom_mask] = 0.0
        # prune
        self._denom = self._denom[keep]
        self._opacity_accum = self._opacity_accum[keep]

        torch.cuda.empty_cache()

    @staticmethod
    def get_remove_duplicates(grid_coords, selected_grid_coords_unique, use_chunk=True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (
                    (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i * chunk_size : (i + 1) * chunk_size, :])
                    .all(-1)
                    .any(-1)
                    .view(-1)
                )
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates

    def on_load_checkpoint(self, module, checkpoint):
        self._offset_gradient_accum = checkpoint["state_dict"]["density_controller._offset_gradient_accum"]
        self._offset_denom = checkpoint["state_dict"]["density_controller._offset_denom"]
        self._opacity_accum = checkpoint["state_dict"]["density_controller._opacity_accum"]
        self._denom = checkpoint["state_dict"]["density_controller._denom"]
        self.register_accums_and_denoms()

    def after_density_changed(self, gaussian_model, optimizers, pl_module):
        self.register_accums_and_denoms()

    @property
    def offset_gradient_accum(self) -> torch.Tensor:
        return self._offset_gradient_accum

    @property
    def opacity_accum(self) -> torch.Tensor:
        return self._opacity_accum

    @property
    def offset_denom(self) -> torch.Tensor:
        return self._offset_denom

    @property
    def denom(self) -> torch.Tensor:
        return self._denom
