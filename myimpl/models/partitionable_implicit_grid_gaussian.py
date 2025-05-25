from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from gsplat.cuda._wrapper import spherical_harmonics
from pytorch3d.transforms import quaternion_multiply

from internal.cameras.cameras import Camera
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


class PartitionableMixin:
    def calculate_implicit_properties(
        self: Union["PartitionableMixin", ImplicitGridGaussianModel],
        viewpoint_camera: Camera,
        appearance_code: Optional[torch.Tensor],
        anchor_mask: Optional[torch.Tensor] = None,
        prog_ratio: Optional[torch.Tensor] = None,
        transition_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if anchor_mask is None:
            anchor_mask = self.get_anchors.new_ones((self.n_anchors,), dtype=torch.bool)
        anchors = self.get_anchors[anchor_mask]
        features = self.get_anchor_features[anchor_mask]
        offsets = self.get_offsets[anchor_mask]
        scalings = self.get_scalings[anchor_mask]
        rotations = self.get_rotations[anchor_mask]
        anchor_partition_ids = self.get_anchor_partition_ids[anchor_mask]

        n_anchors, n_offsets = self.n_anchors, self.n_offsets

        viewdirs = anchors - viewpoint_camera.camera_center
        viewdirs_norm = torch.norm(viewdirs, dim=1, keepdim=True)
        viewdirs = viewdirs / viewdirs_norm

        if self.config.use_feature_bank:
            bank_weight = F.softmax(
                self.forward_by_partition_id(self.get_feature_bank_mlp, anchor_partition_ids, viewdirs), dim=-1
            ).unsqueeze(dim=1)
            features = features.unsqueeze(dim=-1)
            features = (
                features[:, ::4, :1].repeat(1, 4, 1) * bank_weight[:, :, 0:1]
                + features[:, ::2, :1].repeat(1, 2, 1) * bank_weight[:, :, 1:2]
                + features[:, ::1, :1] * bank_weight[:, :, 2:3]
            )
            features = features.squeeze(dim=-1)
        cat_local_view = torch.cat([viewdirs, features], dim=1)

        opacities_offsets = self.forward_by_partition_id(
            self.get_opacity_mlp, anchor_partition_ids, features
        )  # try: remove viewdirs
        opacities = torch.clamp(opacities_offsets, max=1.0).reshape(-1, n_offsets, 1)
        if prog_ratio is not None and transition_mask is not None:
            prog = prog_ratio[anchor_mask]
            transition = transition_mask[anchor_mask]
            prog[~transition] = 1.0
            opacities = opacities * prog
        opacities = opacities.reshape(-1, 1)

        primitive_mask = (opacities > 0.0).view(-1)

        if appearance_code is not None:
            appearance_code = appearance_code.to(cat_local_view).view(1, -1).repeat(self.n_anchors, 1)
            color_input = torch.cat([cat_local_view, appearance_code], dim=-1)
        else:
            color_input = cat_local_view
        colors = self.forward_by_partition_id(self.get_color_mlp, anchor_partition_ids, color_input).reshape(-1, 3)

        scale_rots = self.forward_by_partition_id(self.get_cov_mlp, anchor_partition_ids, cat_local_view).reshape(-1, n_offsets, 7)  # fmt: skip
        scale_rots[..., -4:] = quaternion_multiply(
            rotations.unsqueeze(1),
            self.rotation_activation(scale_rots[..., -4:].clone()),
        )
        scale_rots = scale_rots.reshape(-1, 7)

        concatenated = repeat(torch.cat([anchors, scalings], dim=-1), "n c -> (n k) c", k=n_offsets)
        concatenated = torch.cat([concatenated, offsets.reshape(-1, 3), opacities, colors, scale_rots], dim=-1)
        concatenated_masked = concatenated[primitive_mask]
        (
            _anchors,
            _scalings_offset,
            _scalings_scales,
            _offsets,
            _opacities,
            _colors,
            _scales,
            _rots,
        ) = torch.split(concatenated_masked, [3, 3, 3, 3, 1, 3, self.color_dim, 4], dim=-1)

        xyz = _anchors + _offsets * _scalings_offset
        scales = F.sigmoid(_scales) * _scalings_scales
        rots = self.rotation_activation(_rots)
        if self.config.color_mode == "RGB":
            colors = _colors
        elif self.config.color_mode == "SHs":
            shs = _colors.reshape(-1, int(self.color_dim // 3), 3)
            viewdirs = xyz.detach() - viewpoint_camera.camera_center
            colors = spherical_harmonics(
                self.activate_sh_degree,
                viewdirs,
                shs,
            )
        opacities = _opacities.squeeze()

        return xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask

    def forward_by_partition_id(
        self: "PartitionableImplicitGridGaussianModel",
        mlp: Dict[int, nn.Sequential],
        anchor_partition_ids: torch.Tensor,
        features: torch.Tensor,
    ):
        for layer in reversed(list(mlp.values())[0].layers):
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

    def create_mlps_oldver(self: "PartitionableImplicitGridGaussianModel"):
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

    def create_mlps(self: "PartitionableImplicitGridGaussianModel"):
        self.gaussian_mlps = nn.ModuleDict()
        self.gaussian_mlps["opacity"] = nn.ModuleDict(
            {str(k): self.create_opacity_mlp() for k in self.config.partition_ids},
        )
        self.gaussian_mlps["cov"] = nn.ModuleDict(
            {str(k): self.create_cov_mlp() for k in self.config.partition_ids},
        )
        self.gaussian_mlps["color"] = nn.ModuleDict(
            {str(k): self.create_color_mlp() for k in self.config.partition_ids},
        )
        if self.config.use_feature_bank:
            self.gaussian_mlps["feature_bank"] = nn.ModuleDict(
                {str(k): self.create_feature_bank_mlp() for k in self.config.partition_ids},
            )

    @property
    def masked_anchor_partition_ids(self):
        return self._masked_anchor_partition_ids

    def set_anchor_partition_ids(self, val: torch.Tensor):
        self._masked_anchor_partition_ids = val

    @property
    def get_opacity_mlp(self):
        return self.gaussian_mlps["opacity"]

    @property
    def get_cov_mlp(self):
        return self.gaussian_mlps["cov"]

    @property
    def get_color_mlp(self):
        return self.gaussian_mlps["color"]

    @property
    def get_feature_bank_mlp(self):
        if self.config.use_feature_bank:
            return self.gaussian_mlps["feature_bank"]
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
