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
    in_dim: int = field(default=-1, init=False)

    out_dim: int = field(default=-1, init=False)

    optimization: AdapterOptimizationConfig = field(default_factory=lambda: AdapterOptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "Adapter":
        return Adapter(self)


class Adapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        if self.config.in_dim != self.config.out_dim:
            mat = torch.zeros((self.config.in_dim, self.config.out_dim), dtype=torch.float).normal_(0, 0.02)
            self.weight = nn.Parameter(mat, requires_grad=True)
        else:
            self.weight = None

    def forward(self, x: torch.Tensor):
        """
        x: [H, W, C]
        """
        if self.weight is None:
            return x
        return torch.einsum("...c, cd -> ...d", x, self.weight)
