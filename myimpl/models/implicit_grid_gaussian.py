from dataclasses import dataclass, field

from myimpl.models.grid_gaussians import (
    GridGaussian,
    GridGaussianModel,
    GridOptimizationConfig,
    LoDGridGaussian,
    LoDGridGaussianModel,
    LoDGridOptimizationConfig,
    ScaffoldGaussianMixin,
    ScaffoldGaussianModelMixin,
    ScaffoldOptimizationConfigMixin,
)


@dataclass
class ImplicitGridOptimizationConfig(GridOptimizationConfig, ScaffoldOptimizationConfigMixin):
    pass


@dataclass
class ImplicitGridGaussian(GridGaussian, ScaffoldGaussianMixin):
    optimization: ImplicitGridOptimizationConfig = field(default_factory=lambda: ImplicitGridOptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return ImplicitGridGaussianModel(self)


class ImplicitGridGaussianModel(ScaffoldGaussianModelMixin, GridGaussianModel):
    config: ImplicitGridGaussian


@dataclass
class ImplicitLoDGridOptimizationConfig(LoDGridOptimizationConfig, ScaffoldOptimizationConfigMixin):
    pass


@dataclass
class ImplicitLoDGridGaussian(LoDGridGaussian, ScaffoldGaussianMixin):
    optimization: ImplicitLoDGridOptimizationConfig = field(default_factory=lambda: ImplicitLoDGridOptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return ImplicitLoDGridGaussianModel(self)


class ImplicitLoDGridGaussianModel(ScaffoldGaussianModelMixin, LoDGridGaussianModel):
    config: ImplicitLoDGridGaussian
