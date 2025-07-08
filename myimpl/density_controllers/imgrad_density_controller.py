from dataclasses import dataclass
from typing import Union

import torch

from internal.density_controllers.density_controller import Utils
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)


@dataclass
class ImGradDensityController(VanillaDensityController):
    density_imgrad_threshold: float = 20.0

    def instantiate(self, *args, **kwargs):
        return ImGradDensityControllerImpl(self)


class ImGradDensityControllerImpl(VanillaDensityControllerImpl):
    config: ImGradDensityController

    def _init_state(self, n_gaussians, device):
        super()._init_state(n_gaussians, device)
        xyz_imgrad_accum = torch.zeros((n_gaussians, 1), device=device)
        self.register_buffer("xyz_imgrad_accum", xyz_imgrad_accum, persistent=True)

    def update_states(self, outputs):
        viewspace_point_tensor, visibility_filter, radii = (
            outputs["viewspace_points"],
            outputs["visibility_filter"],
            outputs["radii"],
        )
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        # update states
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
        xys_grad = viewspace_point_tensor.grad
        if self.config.absgrad is True:
            xys_grad = viewspace_point_tensor.absgrad
        self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)

        xys_imgrad = viewspace_point_tensor.imgrad
        self.xyz_imgrad_accum[visibility_filter] += xys_imgrad[visibility_filter].unsqueeze(1)

    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers):
        min_opacity = self.config.cull_opacity_threshold
        prune_extent = self.prune_extent

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        imgrads = self.xyz_imgrad_accum / self.denom
        imgrads[imgrads.isnan()] = 0.0

        # densify
        self._densify_and_clone(grads, imgrads, gaussian_model, optimizers)
        self._densify_and_split(grads, imgrads, gaussian_model, optimizers)

        # prune
        if self.config.cull_by_max_opacity:
            # TODO: re-implement as a new density controller
            prune_mask = torch.logical_and(
                gaussian_model.get_opacity_max() >= 0.0,
                gaussian_model.get_opacity_max() < min_opacity,
            )
            gaussian_model.reset_opacity_max()
        else:
            prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, imgrads, gaussian_model, optimizers):
        grad_threshold = self.config.densify_grad_threshold
        imgrad_threshold = self.config.density_imgrad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        # Extract points that satisfy the gradient condition
        grad_mask = torch.norm(grads, dim=-1) >= grad_threshold
        imgrad_mask = imgrads.squeeze() >= imgrad_threshold
        selected_pts_mask = torch.where(grad_mask | imgrad_mask, True, False)
        # Exclude big Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

    def _densify_and_split(self, grads, imgrads, gaussian_model, optimizers, N=2):
        grad_threshold = self.config.densify_grad_threshold
        imgrad_threshold = self.config.density_imgrad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        padded_imgrad = torch.zeros((n_init_points,), device=device)
        padded_imgrad[: imgrads.shape[0]] = imgrads.squeeze()

        # Extract points that satisfy the gradient condition
        grad_mask = padded_grad >= grad_threshold
        imgrad_mask = padded_imgrad >= imgrad_threshold
        selected_pts_mask = torch.where(grad_mask | imgrad_mask, True, False)
        # Exclude small Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(
                scales,
                dim=1,
            ).values
            > percent_dense * scene_extent,
        )

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),
                    device=device,
                    dtype=torch.bool,
                ),
            )
        )
        self._prune_points(prune_filter, gaussian_model, optimizers)

    def _prune_points(self, mask, gaussian_model, optimizers):
        valid_points_mask = ~mask  # `True` to keep
        new_parameters = Utils.prune_properties(valid_points_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

        # prune states
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_imgrad_accum = self.xyz_imgrad_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
