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
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import inverse_sigmoid
from internal.utils.network_factory import NetworkFactory
from myimpl.model_components.embeddings import (MLP, MLPWithHashEncoding,
                                                MLPWithMixedHashEncoding,
                                                MLPWithSHEncoding,
                                                MLPWithSHEncodingIdentity)
from myimpl.model_components.feature_adapter import AdapterConfig
from myimpl.model_components.hash_grid import (
    HashGridFeatureConfig, HashGridFeatureOptimizationConfig)
from myimpl.models.implicit_grid_gaussian import (
    ImplicitLoDGridGaussian, ImplicitLoDGridGaussianModel,
    ImplicitLoDGridOptimizationConfig)
from myimpl.utils.dataset_utils import SemanticData


@dataclass
class HashGridAssistedGaussianOptimizationConfig(HashGridFeatureOptimizationConfig, ImplicitLoDGridOptimizationConfig):
    pass


@dataclass
class HashGridAssistedGaussian(ImplicitLoDGridGaussian):
    extend_ratio: float = field(default=0.2)

    optimization: HashGridAssistedGaussianOptimizationConfig = field(
        default_factory=lambda: HashGridAssistedGaussianOptimizationConfig()
    )

    hash_grid_feature: HashGridFeatureConfig = field(default_factory=lambda: HashGridFeatureConfig())

    reduced_feature_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig())

    def instantiate(self, *args, **kwargs):
        return HashGridAssistedGaussianModel(self)


class HashGridAssistedGaussianModel(ImplicitLoDGridGaussianModel):
    config: HashGridAssistedGaussian

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_hash_features(self, xyz: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_xyz(xyz)
        mask = ((normalized > 0.0) & (normalized < 1.0)).all(dim=-1, keepdim=True)
        masked = normalized * mask
        return self.get_hash_grid_feature_mlp(masked)

    def compute_anchor_features(self, anchors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hash_features = self.compute_hash_features(anchors)
        hash_features = self.get_feature_adapter_mlp(hash_features.clone().detach())

        anchor_features = self.get_anchor_features[mask]
        anchor_features = self.get_reduced_feature_adapter_mlp(anchor_features)

        return anchor_features + hash_features

    def create_mlps(self):
        """
        1. configure `config.mlp_in_features` first
        2. configure `config.feature_adapter.in_dim` (`out_dim`) first, if feature_adapter is required
        3. configure `config.hash_grid_feature.out_dim` first, if hash_grid_feature is required
        4. configure `config.reduced_feature_adapter.in_dim` (`out_dim`) first, if reduced_feature_adapter is required
        """

        super().create_mlps()

        if self.config.hash_grid_feature.out_dim > 0:
            self.gaussian_mlps["hash_grid_feature"] = self.config.hash_grid_feature.instantiate()
        if self.config.reduced_feature_adapter.in_dim > 0 and self.config.reduced_feature_adapter.out_dim > 0:
            self.gaussian_mlps["reduced_feature_adapter"] = self.config.reduced_feature_adapter.instantiate()

    def get_extra_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        pl_module: lightning.LightningModule = None,
        *args,
        **kwargs,
    ):
        if "feature" in getattr(pl_module, "renderer_output_types", []):
            semantic_dim = -1
            try:
                train_set = pl_module.trainer.datamodule.dataparser_outputs.train_set
                semantic_dim = train_set.extra_data_processor[SemanticData.KEY].semantic_dim
            except:
                pass

            self.config.mlp_in_features = semantic_dim

            # directly match with semantic regularization
            self.config.hash_grid_feature.out_dim = semantic_dim

            # feature adapter: match output hash grid feature with mlp_in_features
            self.config.feature_adapter.in_dim = self.config.hash_grid_feature.out_dim
            self.config.feature_adapter.out_dim = self.config.mlp_in_features

            # reduced feature adapter: match feature_dim with mlp_in_features
            self.config.reduced_feature_adapter.in_dim = self.config.feature_dim
            self.config.reduced_feature_adapter.out_dim = self.config.mlp_in_features

        property_dict = super().get_extra_properties(
            fused_point_cloud=fused_point_cloud, n=n, tensors=tensors, mode=mode, pl_module=pl_module, *args, **kwargs
        )

        if mode == "tensors":  # TODO: mlp structure may be different
            _tensors = tensors["mlps"]
            if "hash_grid_feature" in _tensors:
                self.gaussian_mlps["hash_grid_feature"].load_state_dict(_tensors["hash_grid_feature"])
            if "reduced_feature_adapter" in _tensors:
                self.gaussian_mlps["reduced_feature_adapter"].load_state_dict(_tensors["reduced_feature_adapter"])

        return property_dict

    def training_setup_extra_properties(self, module, *args, **kwargs):
        optimizers, schedulers = super().training_setup_extra_properties(module, *args, **kwargs)

        optimization_config = self.config.optimization
        mlp_optimizer_factory = self.config.optimization.mlp_optimizer
        mlp_scheduler_factory = self.config.optimization.mlp_scheduler

        mlp_l = [
            {
                "params": self.gaussian_mlps["hash_grid_feature"].parameters(),
                "lr": optimization_config.hash_feature_lr_init,
                "name": "hash_grid_feature_mlp",
            },
            {
                "params": self.gaussian_mlps["reduced_feature_adapter"].parameters(),
                "lr": self.config.feature_adapter.optimization.lr_init,
                "name": "reduced_feature_adapter_mlp",
            },
        ]
        mlp_optimizer = mlp_optimizer_factory.instantiate(mlp_l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(mlp_optimizer, module)

        mlp_scheduler = mlp_scheduler_factory.instantiate().get_schedulers(
            mlp_optimizer,
            [
                optimization_config.hash_feature_lr_final,
                self.config.feature_adapter.optimization.lr_final,
            ],
        )

        optimizers.insert(0, mlp_optimizer)
        schedulers.insert(0, mlp_scheduler)
        return optimizers, schedulers

    @property
    def get_hash_grid_feature_mlp(self):
        if "hash_grid_feature" not in self.gaussian_mlps:
            return nn.Identity()
        return self.gaussian_mlps["hash_grid_feature"]

    @property
    def get_reduced_feature_adapter_mlp(self):
        if "reduced_feature_adapter" not in self.gaussian_mlps:
            return nn.Identity()
        return self.gaussian_mlps["reduced_feature_adapter"]
