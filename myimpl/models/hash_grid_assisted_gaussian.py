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
from myimpl.model_components.feature_adapter import AdapterConfig
from myimpl.model_components.hash_feature_grid import (
    HashFeatureGridConfig, HashFeatureGridOptimizationConfig)
from myimpl.models.implicit_grid_gaussian import (
    ImplicitLoDGridGaussian, ImplicitLoDGridGaussianModel,
    ImplicitLoDGridOptimizationConfig)
from myimpl.utils.dataset_utils import SemanticData


@dataclass
class HashGridAssistedGaussianOptimizationConfig(HashFeatureGridOptimizationConfig, ImplicitLoDGridOptimizationConfig):
    pass


@dataclass
class HashGridAssistedGaussian(ImplicitLoDGridGaussian):
    extend_ratio: float = field(default=0.2)

    hash_feature_grid: HashFeatureGridConfig = field(default_factory=lambda: HashFeatureGridConfig())

    reduced_feature_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig())

    optimization: HashGridAssistedGaussianOptimizationConfig = field(
        default_factory=lambda: HashGridAssistedGaussianOptimizationConfig()
    )

    def instantiate(self, *args, **kwargs):
        return HashGridAssistedGaussianModel(self)


class HashGridAssistedGaussianModel(ImplicitLoDGridGaussianModel):
    config: HashGridAssistedGaussian

    def compute_hash_features(self, xyz: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_xyz(xyz)
        mask = ((normalized > 0.0) & (normalized < 1.0)).all(dim=-1, keepdim=True)
        masked = normalized * mask
        return self.get_hash_feature_mlp(masked).float()

    def compute_anchor_features(self, anchors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        anchor_features = self.get_anchor_features[mask]
        hash_features = self.compute_hash_features(anchors).clone().detach()
        # anchor_features = self.get_feature_adapter_mlp(anchor_features)
        anchor_features = self.get_reduced_feature_adapter_mlp(anchor_features)
        hash_features = self.get_feature_adapter_mlp(hash_features)

        return anchor_features + hash_features

    def create_mlps(self, pl_module, feature_dim: Optional[int] = None):
        train_set = pl_module.trainer.datamodule.dataparser_outputs.train_set
        semantic_dim = train_set.extra_data_processor[SemanticData.KEY].dim
        assert semantic_dim > 0, "Semantic data is not available for training."

        # configure hash_feature_grid and feature_adapter to match the new feature_dim
        # self.config.hash_feature_grid.out_dim = semantic_dim
        # self.config.feature_adapter.in_dim = self.config.feature_dim
        # self.config.feature_adapter.out_dim = semantic_dim
        # super().create_mlps(pl_module, feature_dim=semantic_dim)

        self.config.hash_feature_grid.out_dim = self.config.feature_dim  # TODO: maybe semantic dim
        self.config.feature_adapter.in_dim = self.config.feature_dim
        self.config.feature_adapter.out_dim = semantic_dim
        self.config.reduced_feature_adapter.in_dim = self.config.feature_dim
        self.config.reduced_feature_adapter.out_dim = semantic_dim
        super().create_mlps(pl_module, feature_dim=semantic_dim)

        self.gaussian_mlps["hash_feature_mlp"] = self.config.hash_feature_grid.instantiate()
        self.gaussian_mlps["reduced_feature_mlp"] = self.config.reduced_feature_adapter.instantiate()

    def get_extra_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        *args,
        **kwargs,
    ):
        # add temp `anchor_features`
        property_dict = super().get_extra_properties(
            fused_point_cloud=fused_point_cloud, n=n, tensors=tensors, mode=mode, *args, **kwargs
        )

        if mode == "tensors":
            self.gaussian_mlps["hash_feature_mlp"].load_state_dict(tensors["hash_feature_mlp"])
        return property_dict

    def training_setup_extra_properties(self, module, *args, **kwargs):
        optimizers, schedulers = super().training_setup_extra_properties(module, *args, **kwargs)

        optimization_config = self.config.optimization
        mlp_optimizer_factory = self.config.optimization.mlp_optimizer
        mlp_scheduler_factory = self.config.optimization.mlp_scheduler
        mlp_l = [
            {
                "params": self.gaussian_mlps["hash_feature_mlp"].parameters(),
                "lr": self.config.optimization.hash_feature_lr_init,
                "name": "hash_feature_mlp",  # `_mlp` avoid errors when cat tensors to optimizers
            },
            {
                "params": self.gaussian_mlps["reduced_feature_mlp"].parameters(),
                "lr": self.config.feature_adapter.optimization.lr_init,
                "name": "reduced_adapter_mlp",
            },
        ]
        optimizer = mlp_optimizer_factory.instantiate(mlp_l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(optimizer, module)
        scheduler = mlp_scheduler_factory.instantiate().get_schedulers(
            optimizer, [optimization_config.hash_feature_lr_final, self.config.feature_adapter.optimization.lr_final]
        )

        optimizers.append(optimizer)
        schedulers.append(scheduler)

        return optimizers, schedulers

    @property
    def get_hash_feature_mlp(self):
        return self.gaussian_mlps["hash_feature_mlp"]

    @property
    def get_reduced_feature_adapter_mlp(self):
        return self.gaussian_mlps["reduced_feature_mlp"]
