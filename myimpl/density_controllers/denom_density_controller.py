from typing import List

import lightning
import torch

from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from internal.models.vanilla_gaussian import VanillaGaussianModel

__all__ = ["DenomDensityController", "DenomDensityControllerImpl"]


class DenomDensityController(VanillaDensityController):
    success_ratio: float = 0.5

    def instantiate(self, *args, **kwargs):
        return DenomDensityControllerImpl(self)


class DenomDensityControllerImpl(VanillaDensityControllerImpl):
    config: DenomDensityController

    def setup(self, stage: str, pl_module: lightning.LightningModule) -> None:
        super().setup(stage, pl_module)
        if stage == "fit":
            self.prune_denom: torch.Tensor
            self.register_buffer("prune_denom", torch.zeros((pl_module.gaussian_model.n_gaussians,)))

    def _init_state(self, n_gaussians, device):
        super()._init_state(n_gaussians, device)

        if getattr(self, "prune_denom", None) is None:
            self.register_buffer("prune_denom", torch.zeros((n_gaussians,)))
        else:
            n_padding = n_gaussians - self.prune_denom.shape[0]
            self.register_buffer(
                "prune_denom",
                torch.cat([self.prune_denom, torch.zeros((n_padding,), device=self.prune_denom.device)], dim=0),
            )

    def update_states(self, outputs):
        super().update_states(outputs)

        visibility_filter = outputs["visibility_filter"]
        self.prune_denom[visibility_filter] += 1

    def _prune_points(self, mask, gaussian_model, optimizers):
        prune_mask = self.prune_denom > self.config.success_ratio * self.config.opacity_reset_interval
        mask = torch.logical_and(mask, prune_mask)
        self.prune_denom[prune_mask] = 0

        super()._prune_points(mask, gaussian_model, optimizers)
        self.prune_denom = self.prune_denom[~mask]
