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
class AdapterOptimizationConfig:
    lr_init: float = 2e-3

    lr_final: float = 2e-5

    max_steps: int = None


@dataclass
class AdapterConfig:
    dim_out: int = 32

    optimization: AdapterOptimizationConfig = field(default_factory=lambda: AdapterOptimizationConfig())

    def instantiate(self, dim_in, *args, **kwargs) -> "Adapter":
        return Adapter(self, dim_in)


class Adapter(nn.Module):
    def __init__(self, config: AdapterConfig, dim_in: int):
        super().__init__()
        self.config = config

        mat = torch.zeros((dim_in, self.config.dim_out), dtype=torch.float).normal_(0, 0.02)
        self.weight = nn.Parameter(mat, requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        x: [H, W, C]
        """
        out = torch.einsum("...hwc, cd -> ...hwd", x, self.weight)
        return out

    def training_setup(self):
        # fmt: off
        net_optimizer = Adam().instantiate(
            [{
                "params": self.weight,
                "lr": self.config.optimization.lr_init,
                "name": "feature_adapter_mlp",
            }],
            lr=0.0,
        )
        # fmt: on
        net_scheduler = (
            ExponentialDecayScheduler(
                lr_final=self.config.optimization.lr_final,
                max_steps=self.config.optimization.max_steps,
            )
            .instantiate()
            .get_scheduler(optimizer=net_optimizer, lr_init=self.config.optimization.lr_init)
        )

        return [net_optimizer], [net_scheduler]
