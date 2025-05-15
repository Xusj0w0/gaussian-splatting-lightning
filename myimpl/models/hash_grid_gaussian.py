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
from myimpl.model_components.grid_encodings import (MLP, MLPWithHashEncoding,
                                                    MLPWithMixedHashEncoding)
from myimpl.models.implicit_grid_gaussian import (
    ImplicitLoDGridGaussian, ImplicitLoDGridGaussianModel,
    ImplicitLoDGridOptimizationConfig)


@dataclass
class HashLoDGridOptimizationConfig(ImplicitLoDGridOptimizationConfig):
    anchor_features_lr: float = field(init=False)

    hash_feature_lr_init: float = 0.005
    hash_feature_lr_final: float = 0.0005


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

    num_levels: int = 16
    min_res: int = 2 << 4
    max_res: int = 2 << 13
    log2_hashmap_size: int = 19
    features_per_level: int = 4

    use_mixed: bool = True
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
        self.config.hash_grid_feature.out_dim = self.config.feature_dim

        if "anchor_features" in self._names:
            _names = list(self._names)
            _names.remove("anchor_features")
            self._names = tuple(_names)

    def calculate_implicit_properties(
        self,
        viewpoint_camera: Camera,
        appearance_code: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        prog_ratio: Optional[torch.Tensor] = None,
        transition_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if anchor_mask is None:
            anchor_mask = self.get_anchors.new_ones((self.n_anchors,), dtype=torch.bool)
        anchors = self.get_anchors[anchor_mask]
        offsets = self.get_offsets[anchor_mask]
        scalings = self.get_scalings[anchor_mask]
        rotations = self.get_rotations[anchor_mask]
        features = self.compute_hash_features(anchors)

        if kwargs.get("stop_feature_grad", False):
            features = features.clone().detach()

        n_anchors, n_offsets = self.n_anchors, self.n_offsets

        viewdirs = anchors - viewpoint_camera.camera_center
        viewdirs_norm = torch.norm(viewdirs, dim=1, keepdim=True)
        viewdirs = viewdirs / viewdirs_norm

        if self.config.use_feature_bank:
            bank_weight = F.softmax(self.get_feature_bank_mlp(viewdirs), dim=-1).unsqueeze(dim=1)
            features = features.unsqueeze(dim=-1)
            features = (
                features[:, ::4, :1].repeat(1, 4, 1) * bank_weight[:, :, 0:1]
                + features[:, ::2, :1].repeat(1, 2, 1) * bank_weight[:, :, 1:2]
                + features[:, ::1, :1] * bank_weight[:, :, 2:3]
            )
            features = features.squeeze(dim=-1)
        cat_local_view = torch.cat([features, viewdirs], dim=1)

        opacities = self.get_opacity_mlp(features).reshape(-1, n_offsets, 1).clamp(max=1.0)
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
        colors = self.get_color_mlp(color_input).reshape(-1, 3)

        scale_rots = self.get_cov_mlp(cat_local_view).reshape(-1, n_offsets, 7)
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
        colors = _colors
        opacities = _opacities.squeeze()

        return xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask

    def compute_hash_features(self, xyz: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_xyz(xyz)
        mask = ((normalized > 0.0) & (normalized < 1.0)).all(dim=-1, keepdim=True)
        masked = normalized * mask
        return self.get_hash_feature_mlp(masked).float()

    def create_mlps(self):
        super().create_mlps()

        self.gaussian_mlps["hash_feature_mlp"] = self.config.hash_grid_feature.instantiate()

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
        if mode == "tensors" and "anchor_features" not in tensors:
            tensors["anchor_features"] = torch.zeros((1,), dtype=torch.float)
        property_dict = super().get_extra_properties(
            fused_point_cloud=fused_point_cloud, n=n, tensors=tensors, mode=mode, *args, **kwargs
        )
        del property_dict["anchor_features"]

        if mode == "tensors":
            self.gaussian_mlps["hash_feature_mlp"].load_state_dict(tensors["hash_feature_mlp"])
        return property_dict

    def training_setup_extra_properties(self, module, *args, **kwargs):
        if "anchor_features" not in self.gaussians:
            self.gaussians["anchor_features"] = nn.Parameter(torch.zeros((1,)), requires_grad=True)

        [mlp_optimizer, constant_optimizer], [mlp_scheduler] = super().training_setup_extra_properties(
            module, *args, **kwargs
        )

        if "anchor_features" in self.gaussians:
            del self.gaussians["anchor_features"]
            constant_optimizer.param_groups = [
                p for p in constant_optimizer.param_groups if p["name"] != "anchor_features"
            ]

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

        return [mlp_optimizer, optimizer], [mlp_scheduler, scheduler]

    @property
    def get_anchor_features(self):
        return self.gaussian_mlps["hash_feature_mlp"](self.get_anchors)

    @property
    def get_hash_feature_mlp(self):
        return self.gaussian_mlps["hash_feature_mlp"]
