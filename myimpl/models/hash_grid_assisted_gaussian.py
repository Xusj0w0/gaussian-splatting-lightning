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
from myimpl.model_components.feature_adapter import AdapterConfig, FusionConfig
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

    feature_fusion: FusionConfig = field(default_factory=lambda: FusionConfig())

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
        with torch.no_grad():
            hash_features = self.compute_hash_features(anchors)
        anchor_features = self.get_anchor_features[mask]
        return self.get_feature_fusion_mlp(hash_features, anchor_features)

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
        if self.config.feature_fusion.is_valid():
            self.gaussian_mlps["feature_fusion"] = self.config.feature_fusion.instantiate()

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

            # feature adapter: match output hash grid feature with mlp_in_features
            self.config.feature_adapter.in_dim = self.config.hash_grid_feature.out_dim
            self.config.feature_adapter.out_dim = self.config.mlp_in_features

            # feature fusion: match feature_dim with mlp_in_features
            self.config.feature_fusion.hash_feature_dim = self.config.hash_grid_feature.out_dim
            self.config.feature_fusion.residual_feature_dim = self.config.feature_dim
            self.config.feature_fusion.mlp_in_dim = self.config.mlp_in_features

        property_dict = super().get_extra_properties(
            fused_point_cloud=fused_point_cloud, n=n, tensors=tensors, mode=mode, pl_module=pl_module, *args, **kwargs
        )

        if mode == "tensors":  # TODO: mlp structure may be different
            config = tensors.get("config", None)

            _tensors = tensors["properties"]
            anchor_features = _tensors["anchor_features"]
            if anchor_features.shape[-1] <= 0:
                # use hash grid and fusion module
                # feature dim should be the same as mlp_in_features
                if config is not None:
                    self.config.feature_dim = config.mlp_in_features

            _tensors = tensors["mlps"]
            device = next(self.gaussian_mlps["opacity"].parameters()).device
            # hash grid feature
            if "hash_grid_feature" in _tensors:
                try:
                    self.gaussian_mlps["hash_grid_feature"].load_state_dict(_tensors["hash_grid_feature"])
                except:
                    if config is not None:
                        self.config.hash_grid_feature.update_config(config.hash_grid_feature)
                        self.gaussian_mlps["hash_grid_feature"] = self.config.hash_grid_feature.instantiate()
                        self.gaussian_mlps["hash_grid_feature"].load_state_dict(_tensors["hash_grid_feature"])

            # feature fusion
            if "feature_fusion" in _tensors:
                try:
                    self.gaussian_mlps["feature_fusion"].load_state_dict(_tensors["feature_fusion"])
                except:
                    if config is not None:
                        self.config.feature_fusion.update_config(config.feature_fusion)
                        self.gaussian_mlps["feature_fusion"] = self.config.feature_fusion.instantiate()
                        self.gaussian_mlps["feature_fusion"].load_state_dict(_tensors["feature_fusion"])

            self.gaussian_mlps.to(device)

        return property_dict

    def training_setup_extra_properties(self, module, *args, **kwargs):
        optimizers, schedulers = super().training_setup_extra_properties(module, *args, **kwargs)

        optimization_config = self.config.optimization
        mlp_optimizer_factory = self.config.optimization.mlp_optimizer
        mlp_scheduler_factory = self.config.optimization.mlp_scheduler

        # hash grid
        hashgrid_optimizer, hashgrid_scheduler = None, None
        hashgrid = getattr(self, "get_hash_grid_feature_mlp", None)
        if hashgrid is not None:
            hashgrid_optimizer = mlp_optimizer_factory.instantiate(
                [
                    {
                        "params": hashgrid.parameters(),
                        "lr": optimization_config.hash_feature_lr_init,
                        "name": "hash_grid_feature_mlp",
                    }
                ],
                lr=0.0,
            )
            self._add_optimizer_after_backward_hook_if_available(hashgrid_optimizer, module)
            hashgrid_scheduler = mlp_scheduler_factory.instantiate().get_scheduler(
                hashgrid_optimizer, optimization_config.hash_feature_lr_final
            )
        optimizers += [hashgrid_optimizer] if hashgrid_optimizer is not None else []
        schedulers += [hashgrid_scheduler] if hashgrid_scheduler is not None else []

        # fusion
        fusion_optimizer, fusion_scheduler = None, None
        fusion = getattr(self, "get_feature_fusion_mlp", None)
        if fusion is not None:
            fusion_optimizer, fusion_scheduler = fusion.training_setup(module)
        if fusion_optimizer is not None:
            self._add_optimizer_after_backward_hook_if_available(fusion_optimizer, module)
        optimizers += [fusion_optimizer] if fusion_optimizer is not None else []
        schedulers += [fusion_scheduler] if fusion_scheduler is not None else []

        return optimizers, schedulers

    # def load_state_dict(self, state_dict, strict=True):
    #     return super().load_state_dict(state_dict, strict=False)

    @property
    def get_hash_grid_feature_mlp(self):
        if "hash_grid_feature" not in self.gaussian_mlps:
            return None
        return self.gaussian_mlps["hash_grid_feature"]

    @property
    def get_feature_fusion_mlp(self):
        if "feature_fusion" not in self.gaussian_mlps:
            return None
        return self.gaussian_mlps["feature_fusion"]
