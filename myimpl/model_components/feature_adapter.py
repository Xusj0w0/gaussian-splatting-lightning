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
    weight_lr: float = 2e-3

    # embedding_lr_init: float = 2e-3
    # embedding_lr_final: float = 2e-4

    max_steps: int = None


@dataclass
class AdapterConfig:
    network_n_layers: int = 2

    network_hidden_dim: int = 64

    # embedding_dim: int = 0

    optimization: AdapterOptimizationConfig = field(default_factory=lambda: AdapterOptimizationConfig())

    def instantiate(self, render_feature_dim, render_feature_size, gt_feature_shape, *args, **kwargs) -> "Adapter":
        return Adapter(
            render_feature_dim=render_feature_dim,
            render_feature_size=render_feature_size,
            gt_feature_shape=gt_feature_shape,
            config=self,
        )


class Adapter(nn.Module):
    def __init__(
        self, render_feature_dim: int, render_feature_size: int, gt_feature_shape: List[int], config: AdapterConfig
    ):
        super().__init__()
        self.config = config

        mat = torch.zeros((render_feature_dim, gt_feature_shape[-1]), dtype=torch.float).normal_(0, 0.02)
        self.weight = nn.Parameter(mat, requires_grad=True)
        self.output_size = gt_feature_shape[:2]

    def forward(self, x: torch.Tensor):
        """
        x: [H, W, C]
        """
        out = torch.einsum("h w c, c d -> h w d", x, self.weight)
        # fmt: off
        out = F.interpolate(
            out.permute(2, 0, 1).unsqueeze(0), size=self.output_size, mode="bicubic", align_corners=True
        ).squeeze(0).permute(1, 2, 0)
        # fmt: on
        return out

    def training_setup(self):
        # fmt: off
        net_optimizer = Adam().instantiate(
            [{
                "params": self.weight,
                "lr": self.config.optimization.weight_lr,
                "name": "adapter_network",
            }],
            lr=0.0,
        )
        # fmt: on
        # net_scheduler = (
        #     ExponentialDecayScheduler(
        #         lr_final=self.config.optimization.network_lr_final,
        #         max_steps=self.config.optimization.max_steps,
        #     )
        #     .instantiate()
        #     .get_scheduler(optimizer=net_optimizer, lr_init=self.config.optimization.weight_lr)
        # )

        return [net_optimizer], []
