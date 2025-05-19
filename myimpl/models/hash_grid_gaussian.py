from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from gsplat.cuda._wrapper import spherical_harmonics
from pytorch3d.transforms import quaternion_multiply
from torch import nn

from internal.cameras.cameras import Camera
from myimpl.model_components.hash_feature_grid import (
    HashFeatureGridConfig, HashFeatureGridOptimizationConfig)
from myimpl.models.implicit_grid_gaussian import (
    ImplicitLoDGridGaussian, ImplicitLoDGridGaussianModel,
    ImplicitLoDGridOptimizationConfig)


@dataclass
class HashGridGaussianOptimizationConfig(HashFeatureGridOptimizationConfig, ImplicitLoDGridOptimizationConfig):
    pass


@dataclass
class HashGridGaussian(ImplicitLoDGridGaussian):
    hash_feature_grid: HashFeatureGridConfig = field(default_factory=lambda: HashFeatureGridConfig())

    optimization: HashGridGaussianOptimizationConfig = field(
        default_factory=lambda: HashGridGaussianOptimizationConfig()
    )

    def instantiate(self, *args, **kwargs):
        return HashGridGaussianModel(self)


class HashGridGaussianModel(ImplicitLoDGridGaussianModel):
    """
    replace `anchor_features` with hash grid features
    """

    config: HashGridGaussian

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.hash_feature_grid.out_dim = self.config.feature_dim

        if "anchor_features" in self._names:
            _names = list(self._names)
            _names.remove("anchor_features")
            self._names = tuple(_names)

    def compute_hash_features(self, anchors: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_xyz(anchors)
        mask = ((normalized > 0) & (normalized < 1.0)).all(dim=-1, keepdim=True)
        masked = normalized * mask
        return self.get_hash_feature_mlp(masked).float()

    def compute_anchor_features(self, anchors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hash_features = self.compute_hash_features(anchors)
        return hash_features

    def create_mlps(self):
        self.config.hash_feature_grid.out_dim = self.config.feature_dim
        super().create_mlps()
        self.gaussian_mlps["hash_feature_mlp"] = self.config.hash_feature_grid.instantiate()

    def get_extra_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        *args,
        **kwargs,
    ):
        if mode == "tensors" and "anchor_features" not in tensors:
            tensors["anchor_features"] = torch.zeros((1,), dtype=torch.float)
        property_dict = super().get_extra_properties(
            fused_point_cloud=fused_point_cloud, n=n, tensors=tensors, mode=mode, *args, **kwargs
        )
        del property_dict["anchor_features"]

        if mode == "tensors":
            self.gaussian_mlps["hash_feature_mlp"].load_state_dict(tensors["hash_feature_mlp"])
        return property_dict

    def training_setup_extra_properties(self, module, *args, **kwargs):
        if "anchor_features" not in self.gaussians:
            self.gaussians["anchor_features"] = nn.Parameter(torch.zeros((1,)), requires_grad=True)
        optimizers, schedulers = super().training_setup_extra_properties(module, *args, **kwargs)
        if "anchor_features" in self.gaussians:
            del self.gaussians["anchor_features"]
            ids = []
            for i, p in enumerate(optimizers[-1].param_groups):
                if p["name"] == "anchor_features":
                    ids.append(i)
            for i in ids:
                optimizers[-1].param_groups = [p for i, p in enumerate(optimizers[-1].param_groups) if i not in ids]
                schedulers[-1].lr_lambdas.pop(i)

        optimization_config = self.config.optimization
        mlp_optimizer_factory = self.config.optimization.mlp_optimizer
        mlp_scheduler_factory = self.config.optimization.mlp_scheduler
        mlp_l = [
            {
                "params": self.gaussian_mlps["hash_feature_mlp"].parameters(),
                "lr": self.config.optimization.hash_feature_lr_init,
                "name": "hash_encoding_mlp",  # `_mlp` avoid errors when cat tensors to optimizers
            },
        ]
        optimizer = mlp_optimizer_factory.instantiate(mlp_l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(optimizer, module)
        scheduler = mlp_scheduler_factory.instantiate().get_scheduler(
            optimizer, optimization_config.hash_feature_lr_final
        )

        optimizers.insert(0, optimizer)
        schedulers.insert(0, scheduler)
        return optimizers, schedulers

    @property
    def get_hash_feature_mlp(self) -> nn.Module:
        return self.gaussian_mlps["hash_feature_mlp"]

    @property
    def get_anchor_features(self):
        return None
