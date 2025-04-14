from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn

from internal.utils.network_factory import NetworkFactory
from myimpl.models.implicit_grid_gaussian import (ImplicitGridGaussian,
                                                  ImplicitGridGaussianModel,
                                                  ImplicitLoDGridGaussian,
                                                  ImplicitLoDGridGaussianModel)

__all__ = [
    "PartitionableImplicitGridGaussian",
    "PartitionableImplicitGridGaussianModel",
    "PartitionableImplicitLoDGridGaussian",
    "PartitionableImplicitLoDGridGaussianModel",
]


class PartitionableMLPWrapper:
    def __init__(self, pc: "PartitionableImplicitGridGaussianModel", mlp_key: str):
        self._pc = pc
        self._mlp_key = mlp_key

    def __call__(self, features: torch.Tensor):
        mlp = self._pc.gaussian_mlps[self._mlp_key]
        anchor_partition_ids = self._pc.masked_anchor_partition_ids

        for layer in reversed(list(mlp.values())[0]):
            if isinstance(layer, nn.Linear):
                dim_out = layer.out_features
                break
        output = features.new_zeros((features.shape[0], dim_out))
        unique_ids = torch.unique(anchor_partition_ids)
        for i in unique_ids:
            mask = anchor_partition_ids == i
            output[mask] = mlp[str(i.item())](features[mask])
        return output


class PartitionableMixin:

    def forward_by_partition_id(
        self: "PartitionableImplicitGridGaussianModel",
        mlp: Dict[int, nn.Sequential],
        anchor_partition_ids: torch.Tensor,
        features: torch.Tensor,
    ):
        for layer in reversed(list(mlp.values())[0]):
            if isinstance(layer, nn.Linear):
                dim_out = layer.out_features
                break
        output = features.new_zeros((features.shape[0], dim_out))
        unique_ids = torch.unique(anchor_partition_ids)
        for i in unique_ids:
            mask = anchor_partition_ids == i
            output[mask] = mlp[str(i.item())](features[mask])
        return output

    def setup_from_number(self, n, *args, **kwargs):
        self.register_buffer("_anchor_partition_ids", torch.zeros((n,), dtype=torch.int))
        super().setup_from_number(n, *args, **kwargs)

    @property
    def get_anchor_partition_ids(self: "PartitionableImplicitGridGaussianModel"):
        self._anchor_partition_ids: torch.Tensor
        return self._anchor_partition_ids

    def create_mlps(self: "PartitionableImplicitGridGaussianModel"):
        self.gaussian_mlps = nn.ModuleDict()
        self.gaussian_mlps["opacity"] = nn.ModuleDict(
            {
                str(k): NetworkFactory(tcnn=self.config.tcnn).get_network(
                    n_input_dims=self.config.feature_dim,  # try: remove viewdirs + self.config.view_dim,
                    n_output_dims=self.n_offsets,
                    n_layers=self.config.mlp_n_layers,
                    n_neurons=self.config.feature_dim,
                    activation="ReLU",
                    output_activation="Tanh",
                )
                for k in self.config.partition_ids
            }
        )
        self.gaussian_mlps["cov"] = nn.ModuleDict(
            {
                str(k): NetworkFactory(tcnn=self.config.tcnn).get_network(
                    n_input_dims=self.config.feature_dim + self.config.view_dim,
                    n_output_dims=7 * self.n_offsets,
                    n_layers=self.config.mlp_n_layers,
                    n_neurons=self.config.feature_dim,
                    activation="ReLU",
                    output_activation="None",
                )
                for k in self.config.partition_ids
            }
        )
        self.gaussian_mlps["color"] = nn.ModuleDict(
            {
                str(k): NetworkFactory(tcnn=self.config.tcnn).get_network(
                    n_input_dims=self.config.feature_dim
                    + self.config.view_dim
                    + self.config.n_appearance_embedding_dims,
                    n_output_dims=self.color_dim * self.n_offsets,
                    n_layers=self.config.mlp_n_layers,
                    n_neurons=self.config.feature_dim,
                    activation="ReLU",
                    output_activation="Sigmoid",
                )
                for k in self.config.partition_ids
            }
        )
        if self.config.use_feature_bank:
            self.gaussian_mlps["feature_bank"] = nn.ModuleDict(
                {
                    str(k): NetworkFactory(tcnn=self.config.tcnn).get_network(
                        n_input_dims=self.config.view_dim,
                        n_output_dims=3,
                        n_layers=2,
                        n_neurons=self.config.feature_dim,
                        activation="ReLU",
                        output_activation="None",
                    )
                    for k in self.config.partition_ids
                }
            )

    @property
    def masked_anchor_partition_ids(self):
        return self._masked_anchor_partition_ids

    def set_anchor_partition_ids(self, val: torch.Tensor):
        self._masked_anchor_partition_ids = val

    @property
    def get_opacity_mlp(self):
        if getattr(self, "_partitionable_opacity_mlp", None) is None:
            self._partitionable_opacity_mlp = PartitionableMLPWrapper(self, "opacity")
        return self._partitionable_opacity_mlp

    @property
    def get_cov_mlp(self):
        if getattr(self, "_partitionable_cov_mlp", None) is None:
            self._partitionable_cov_mlp = PartitionableMLPWrapper(self, "cov")
        return self._partitionable_cov_mlp

    @property
    def get_color_mlp(self):
        if getattr(self, "_partitionable_color_mlp", None) is None:
            self._partitionable_color_mlp = PartitionableMLPWrapper(self, "color")
        return self._partitionable_color_mlp

    @property
    def get_feature_bank_mlp(self):
        if self.config.use_feature_bank:
            if getattr(self, "_partitionable_feature_bank_mlp", None) is None:
                self._partitionable_feature_bank_mlp = PartitionableMLPWrapper(self, "feature_bank")
            return self._partitionable_feature_bank_mlp
        else:
            raise ValueError("Feature bank not available")


@dataclass
class PartitionableImplicitGridGaussian(ImplicitGridGaussian):
    partition_ids: List[int] = field(default_factory=lambda: [])

    def instantiate(self, *args, **kwargs):
        return PartitionableImplicitGridGaussianModel(self)


class PartitionableImplicitGridGaussianModel(PartitionableMixin, ImplicitGridGaussianModel):
    config: PartitionableImplicitGridGaussian

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer_names = tuple(list(self._buffer_names), "_anchor_partition_ids")


@dataclass
class PartitionableImplicitLoDGridGaussian(ImplicitLoDGridGaussian):
    partition_ids: List[int] = field(default_factory=lambda: [])

    def instantiate(self, *args, **kwargs):
        return PartitionableImplicitLoDGridGaussianModel(self)


class PartitionableImplicitLoDGridGaussianModel(PartitionableMixin, ImplicitLoDGridGaussianModel):
    config: PartitionableImplicitLoDGridGaussian

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer_names = tuple(list(self._buffer_names) + ["_anchor_partition_ids"])
