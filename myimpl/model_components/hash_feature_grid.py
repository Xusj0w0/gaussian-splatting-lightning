from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import nn

from internal.optimizers import Adam
from internal.schedulers import ExponentialDecayScheduler
from myimpl.model_components.embeddings import (MLP, MLPWithHashEncoding,
                                                MLPWithMixedHashEncoding)
from myimpl.models.implicit_grid_gaussian import \
    ImplicitLoDGridOptimizationConfig


@dataclass
class HashFeatureGridOptimizationConfig:
    hash_feature_lr_init: float = 5e-3
    hash_feature_lr_final: float = 5e-5


@dataclass
class HashFeatureGridConfig:
    """
    hash table size = 2^log2_haspmap_size * num_levels * features_per_level

    for example:
        - log2_hashmap_size=23
        - num_levels=16
        - features_per_level=4

    hash table size = 2^21 * 16 * 4 = 2M * 64 = 128M

    A scene containing 1e7 anchors with 32-dim features, param number = 1e7 * 32 = 320M
    """

    num_levels: int = 8
    min_res: int = 2 << 4
    max_res: int = 2 << 11
    log2_hashmap_size: int = 15
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
