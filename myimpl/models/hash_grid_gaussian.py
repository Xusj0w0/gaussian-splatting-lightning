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
from myimpl.models.implicit_grid_gaussian import (
    ImplicitLoDGridGaussian, ImplicitLoDGridGaussianModel,
    ImplicitLoDGridOptimizationConfig)


@dataclass
class HashLoDGridOptimizationConfig(ImplicitLoDGridOptimizationConfig):
    hash_feature_lr_init: float = 5e-3
    hash_feature_lr_final: float = 5e-5


@dataclass
class HashGridFeatureConfig:
    """
    hash table size = 2^log2_haspmap_size * num_levels * features_per_level

    for example:
        - log2_hashmap_size=23
        - num_levels=16
        - features_per_level=4

    hash table size = 2^21 * 16 * 4 = 2M * 64 = 128M

    A scene containing 1e7 anchors with 32-dim features, param number = 1e7 * 32 = 320M
    """

    num_levels: int = 12
    min_res: int = 2 << 4
    max_res: int = 2 << 11
    log2_hashmap_size: int = 18
    features_per_level: int = 4

    use_mixed: bool = False
    num_levels_2d: int = 8
    min_res_2d: int = 2 << 8
    max_res_2d: int = 2 << 15
    log2_hashmap_size_2d: int = 15
    features_per_level_2d: int = 4

    tcnn: bool = True
    hash_init_scale: float = 1.0
    interpolation: Optional[Literal["Linear", "Nearest", "Smoothstep"]] = "Linear"

    num_layers: int = 2
    layer_width: int = 64
    out_dim: int = -1
    activation: Literal["ReLU", "Sigmoid", "Tanh", "None"] = "ReLU"
    out_activation: Literal["ReLU", "Sigmoid", "Tanh", "None"] = "None"

    def instantiate(self, *args, **kwargs):
        params = {
            "num_levels": self.num_levels,
            "min_res": self.min_res,
            "max_res": self.max_res,
            "log2_hashmap_size": self.log2_hashmap_size,
            "features_per_level": self.features_per_level,
            "hash_init_scale": self.hash_init_scale,
            "interpolation": self.interpolation,
            "num_layers": self.num_layers,
            "layer_width": self.layer_width,
            "out_dim": self.out_dim,
            "activation": self.activation_str_to_nn_module(self.activation),
            "out_activation": self.activation_str_to_nn_module(self.out_activation),
            "implementation": "tcnn" if self.tcnn else "torch",
        }
        if self.use_mixed:
            return MLPWithMixedHashEncoding(
                num_levels_2d=self.num_levels_2d,
                min_res_2d=self.min_res_2d,
                max_res_2d=self.max_res_2d,
                log2_hashmap_size_2d=self.log2_hashmap_size_2d,
                features_per_level_2d=self.features_per_level_2d,
                **params,
            )
        else:
            return MLPWithHashEncoding(**params)

    def activation_str_to_nn_module(self, activation: str) -> nn.Module:
        if activation == "ReLU":
            return nn.ReLU()
        elif activation == "Sigmoid":
            return nn.Sigmoid()
        elif activation == "Tanh":
            return nn.Tanh()
        elif activation == "None":
            return None
        else:
            raise ValueError(f"Unknown activation function: {activation}")


@dataclass
class HashLoDGridGaussian(ImplicitLoDGridGaussian):
    extend_ratio: float = field(default=0.2)

    optimization: HashLoDGridOptimizationConfig = field(default_factory=lambda: HashLoDGridOptimizationConfig())

    hash_grid_feature: HashGridFeatureConfig = field(default_factory=lambda: HashGridFeatureConfig())

    def instantiate(self, *args, **kwargs):
        return HashLoDGridGaussianModel(self, *args, **kwargs)


class HashLoDGridGaussianModel(ImplicitLoDGridGaussianModel):
    config: HashLoDGridGaussian

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_hash_features(self, xyz: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_xyz(xyz)
        mask = ((normalized > 0.0) & (normalized < 1.0)).all(dim=-1, keepdim=True)
        masked = normalized * mask
        return self.get_hash_feature_mlp(masked)

    def compute_anchor_features(self, anchors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hash_feature = self.compute_hash_features(anchors)
        anchor_features = self.get_anchor_features[mask]
        feature_adapter = self.get_feature_adapter_mlp
        if feature_adapter is not None:
            hash_feature = feature_adapter(hash_feature.clone().detach())

        return hash_feature + anchor_features

    def create_mlps(self):
        self.gaussian_mlps = nn.ModuleDict()
        # create mlps

        # self.config.hash_grid_feature.out_dim = self.config.feature_dim
        # self.gaussian_mlps["feature_adapter"] = self.config.feature_adapter.instantiate(self.config.feature_dim)
        # self.gaussian_mlps["hash_feature_mlp"] = self.config.hash_grid_feature.instantiate()

        self.config.hash_grid_feature.out_dim = self.config.feature_adapter.out_dim
        self.config.feature_adapter.out_dim = self.config.feature_dim
        self.gaussian_mlps["hash_feature_mlp"] = self.config.hash_grid_feature.instantiate()
        self.gaussian_mlps["feature_adapter"] = self.config.feature_adapter.instantiate(
            self.config.hash_grid_feature.out_dim
        )

        feature_dim = self.config.feature_dim  #  + self.config.feature_adapter.dim_out

        # opacity: return 1*n_offsets
        self.gaussian_mlps["opacity"] = MLP(
            in_dim=feature_dim,
            out_dim=self.n_offsets,
            num_layers=self.config.mlp_n_layers,
            layer_width=self.config.hidden_dim,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            implementation="tcnn" if self.config.tcnn else "torch",
        )

        # cov: return 7*n_offsets
        # 3 for scales, multiply with anchor-level scales to get gaussian scales
        # 4 for rotations
        if self.config.view_sh_level > 0:
            cov_mlp = MLPWithSHEncodingIdentity(
                identity_dim=feature_dim,
                levels=self.config.view_sh_level,
                out_dim=7 * self.n_offsets,
                num_layers=self.config.mlp_n_layers,
                layer_width=self.config.hidden_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="tcnn" if self.config.tcnn else "torch",
            )
        else:
            cov_mlp = MLP(
                in_dim=self.config.view_dim + feature_dim,
                out_dim=7 * self.n_offsets,
                num_layers=self.config.mlp_n_layers,
                layer_width=self.config.hidden_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="tcnn" if self.config.tcnn else "torch",
            )
        self.gaussian_mlps["cov"] = cov_mlp

        # color: return 3*n_offsets
        if self.config.view_sh_level > 0:
            color_mlp = MLPWithSHEncodingIdentity(
                identity_dim=feature_dim + self.config.n_appearance_embedding_dims,
                levels=self.config.view_sh_level,
                out_dim=self.color_dim * self.n_offsets,
                num_layers=self.config.mlp_n_layers,
                layer_width=self.config.hidden_dim,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="tcnn" if self.config.tcnn else "torch",
            )
        else:
            color_mlp = MLP(
                in_dim=self.config.view_dim + feature_dim + self.config.n_appearance_embedding_dims,
                out_dim=self.color_dim * self.n_offsets,
                num_layers=self.config.mlp_n_layers,
                layer_width=self.config.hidden_dim,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="tcnn" if self.config.tcnn else "torch",
            )
        self.gaussian_mlps["color"] = color_mlp

        if self.config.use_feature_bank:
            # feature_bank: return 3*n_offsets
            if self.config.view_sh_level > 0:
                feature_bank = MLPWithSHEncoding(
                    levels=self.config.view_sh_level,
                    out_dim=3,
                    num_layers=self.config.mlp_n_layers,
                    layer_width=self.config.hidden_dim,
                    activation=nn.ReLU(),
                    out_activation=None,
                )
            else:
                feature_bank = MLP(
                    in_dim=self.config.view_dim,
                    out_dim=3,
                    num_layers=self.config.mlp_n_layers,
                    layer_width=self.config.hidden_dim,
                    activation=nn.ReLU(),
                    out_activation=None,
                    implementation="tcnn" if self.config.tcnn else "torch",
                )
            self.gaussian_mlps["feature_bank"] = feature_bank

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
                "name": "hash_encoding_mlp",  # `_mlp` avoid errors when cat tensors to optimizers
            },
        ]
        optimizer = mlp_optimizer_factory.instantiate(mlp_l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(optimizer, module)
        scheduler = mlp_scheduler_factory.instantiate().get_scheduler(
            optimizer, optimization_config.hash_feature_lr_final
        )

        optimizers.append(optimizer)
        schedulers.append(scheduler)

        return optimizers, schedulers

    @property
    def get_hash_feature_mlp(self):
        return self.gaussian_mlps["hash_feature_mlp"]
