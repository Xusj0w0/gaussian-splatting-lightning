from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from internal.optimizers import Adam
from internal.schedulers import ExponentialDecayScheduler


def _build_conv_net(dim_in, dim_out, dim_h, n_layers, size_in: int = 256, size_out: int = 64):
    assert size_in % size_out == 0, "size_out must be divisible by size_in"
    scale = size_in // size_out

    layers = [nn.Conv2d(dim_in, dim_h, kernel_size=1, stride=1, padding=0), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.append(nn.Conv2d(dim_h, dim_h, kernel_size=1, stride=1, padding=0))
        layers.append(nn.ReLU())
    layers.append(nn.Conv2d(dim_h, dim_out, kernel_size=1, stride=1, padding=0))
    if scale > 1:
        layers.append(nn.MaxPool2d(kernel_size=scale, stride=scale, padding=0))
    return nn.Sequential(*layers)


@dataclass
class OptimizationConfig:
    lr_init: float = 5e-3

    lr_final: float = 5e-5

    max_steps: int = None


@dataclass
class AdapterConfig:
    in_dim: int = field(default=-1, init=False)

    out_dim: int = field(default=-1, init=False)

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "Adapter":
        return Adapter(self)

    def update_config(self, config: "AdapterConfig"):
        self.in_dim = config.in_dim
        self.out_dim = config.out_dim


class Adapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        self.weight = None
        mat = torch.zeros((self.config.in_dim, self.config.out_dim), dtype=torch.float).normal_(0, 0.02)
        self.weight = nn.Parameter(mat, requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        x: [H, W, C]
        """
        if self.config.in_dim == self.config.out_dim:
            return x
        return torch.einsum("...c, cd -> ...d", x, self.weight)

    def training_setup(self, module):
        if self.weight is None:
            return None, None

        if self.config.optimization.max_steps is None:
            self.config.optimization.max_steps = module.trainer.max_steps
        param_group = [
            {
                "params": self.parameters(),
                "lr": self.config.optimization.lr_init,
                "name": "feature_adapter_mlp",
            }
        ]
        optimizer = Adam().instantiate(param_group, lr=0.0, eps=1e-15)
        scheduler = (
            ExponentialDecayScheduler(
                lr_final=self.config.optimization.lr_final, max_steps=self.config.optimization.max_steps
            )
            .instantiate()
            .get_scheduler(optimizer, lr_init=self.config.optimization.lr_init)
        )
        return optimizer, scheduler


@dataclass
class FusionConfig:
    hash_feature_dim: int = field(default=-1, init=False)

    residual_feature_dim: int = field(default=-1, init=False)

    mlp_in_dim: int = field(default=-1, init=False)

    fix_hash_weight: bool = False

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "Fusion":
        return Fusion(self)

    def is_valid(self):
        return self.hash_feature_dim > 0 and self.residual_feature_dim > 0 and self.mlp_in_dim > 0

    def update_config(self, config: "FusionConfig"):
        self.hash_feature_dim = config.hash_feature_dim
        self.residual_feature_dim = config.residual_feature_dim
        self.mlp_in_dim = config.mlp_in_dim


class Fusion(nn.Module):
    def __init__(self, config: FusionConfig):
        assert config.is_valid(), "hash_feature_dim, residual_feature_dim and mlp_in_dim must be greater than 0"
        super().__init__()
        self.config = config

        hash_mat = torch.zeros((self.config.hash_feature_dim, self.config.mlp_in_dim)).normal_(0, 0.02)
        self.weight_hash = nn.Parameter(hash_mat, requires_grad=not self.config.fix_hash_weight)
        residual_mat = torch.zeros((self.config.residual_feature_dim, self.config.mlp_in_dim)).normal_(0, 0.02)
        self.weight_res = nn.Parameter(residual_mat, requires_grad=True)

    def forward(self, hash_feature: torch.Tensor, residual: torch.Tensor):
        hash_feature = torch.einsum("...c, cd -> ...d", hash_feature, self.weight_hash)
        if self.config.residual_feature_dim != self.config.mlp_in_dim:
            residual = torch.einsum("...c, cd -> ...d", residual, self.weight_res)
        return hash_feature + residual

    def training_setup(self, module):
        if self.config.optimization.max_steps is None:
            self.config.optimization.max_steps = module.trainer.max_steps
        param_group = [
            {
                "params": [p for p in self.parameters() if p.requires_grad],
                "lr": self.config.optimization.lr_init,
                "name": "feature_fusion_mlp",
            }
        ]
        optimizer = Adam().instantiate(param_group, lr=0.0, eps=1e-15)
        scheduler = (
            ExponentialDecayScheduler(
                lr_final=self.config.optimization.lr_final, max_steps=self.config.optimization.max_steps
            )
            .instantiate()
            .get_scheduler(optimizer, lr_init=self.config.optimization.lr_init)
        )
        return optimizer, scheduler
