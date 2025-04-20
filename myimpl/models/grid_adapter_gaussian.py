from dataclasses import dataclass, field

import torch

from myimpl.model_components.feature_adapter import AdapterConfig
from myimpl.models.implicit_grid_gaussian import (
    ImplicitGridGaussian,
    ImplicitGridGaussianModel,
    ImplicitGridOptimizationConfig,
    ImplicitLoDGridGaussian,
    ImplicitLoDGridGaussianModel,
    ImplicitLoDGridOptimizationConfig,
)


class GridAdapterGaussianModelMixin:
    def create_mlps(self):
        super().create_mlps()

        self.gaussian_mlps["feature_adapter_mlp"] = self.config.feature_adapter.instantiate(self.config.feature_dim)

    def get_extra_properties(self, fused_point_cloud=None, n=None, tensors=None, mode="pcd", *args, **kwargs):
        properties = super().get_extra_properties(fused_point_cloud, n, tensors, mode, *args, **kwargs)

        if mode == "tensors":
            if "feature_adapter_mlp" in tensors:
                self.gaussian_mlps["feature_adapter_mlp"].load_state_dict(tensors["feature_adapter_mlp"])
        return properties

    def training_setup_extra_properties(self, module, *args, **kwargs):
        [mlp_optimizer, constant_lr_optimizer], [mlp_scheduler] = super().training_setup_extra_properties(module, *args, **kwargs)

        if self.config.feature_adapter.optimization.max_steps is None:
            self.config.feature_adapter.optimization.max_steps = module.trainer.max_steps

        [adapter_optimizer], [adapter_scheduler] = self.gaussian_mlps["feature_adapter_mlp"].training_setup()

        return [mlp_optimizer, adapter_optimizer, constant_lr_optimizer], [mlp_scheduler, adapter_scheduler]

    @property
    def get_feature_adapter(self):
        return self.gaussian_mlps["feature_adapter_mlp"]


@dataclass
class GridAdapterGaussian(ImplicitGridGaussian):
    feature_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig())

    def instantiate(self, *args, **kwargs):
        return GridAdapterGaussianModel(self)


class GridAdapterGaussianModel(GridAdapterGaussianModelMixin, ImplicitGridGaussianModel):
    config: GridAdapterGaussian
    pass


@dataclass
class LoDGridAdapterGaussian(ImplicitLoDGridGaussian):
    feature_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig())

    def instantiate(self, *args, **kwargs):
        return LoDGridAdapterGaussianModel(self)


class LoDGridAdapterGaussianModel(GridAdapterGaussianModelMixin, ImplicitLoDGridGaussianModel):
    config: LoDGridAdapterGaussian
    pass
