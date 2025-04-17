from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from pytorch3d.transforms import quaternion_multiply

from internal.cameras.cameras import Camera
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from myimpl.models.grid_gaussians import (GridGaussian, GridGaussianModel,
                                          GridOptimizationConfig,
                                          LoDGridGaussian,
                                          LoDGridGaussianModel,
                                          LoDGridOptimizationConfig,
                                          ScaffoldGaussianMixin,
                                          ScaffoldGaussianModelMixin,
                                          ScaffoldOptimizationConfigMixin)
from myimpl.models.grid_gaussians.base import GridGaussianModelBase

__all__ = [
    "ImplicitGridGaussian",
    "ImplicitGridGaussianModel",
    "ImplicitLoDGridGaussian",
    "ImplicitLoDGridGaussianModel",
]


@dataclass
class RefinedScaffoldOptimizationConfigMixin(ScaffoldOptimizationConfigMixin):
    semantic_features_lr: float = 0.0075
    anchor_features_lr: float = 0.01


class RefinedScaffoldGaussianModelMixin(ScaffoldGaussianModelMixin):
    _extra_property_names: List[str] = ["anchor_features", "semantic_features"]

    def calculate_implicit_properties(
        self: Union["RefinedScaffoldGaussianModelMixin", GridGaussianModelBase],
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
        features = self.get_semantic_features[anchor_mask].clone().detach() + self.get_anchor_features[anchor_mask]
        offsets = self.get_offsets[anchor_mask]
        scalings = self.get_scalings[anchor_mask]
        rotations = self.get_rotations[anchor_mask]

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

        opacities_offsets = self.get_opacity_mlp(features).reshape(-1, n_offsets, 1)  # try: remove viewdirs
        opacities = torch.clamp(opacities_offsets, max=1.0)
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

    def get_extra_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        *args,
        **kwargs,
    ):
        self.create_mlps()
        if mode == "pcd":
            assert fused_point_cloud is not None
            n_anchors = fused_point_cloud.shape[0]
            # anchor_features = torch.zeros((n_anchors, self.config.feature_dim), dtype=torch.float)
            semantic_features = torch.normal(0.0, 0.02, (n_anchors, self.config.feature_dim), dtype=torch.float)
            anchor_features = torch.zeros((n_anchors, self.config.feature_dim), dtype=torch.float)

        elif mode == "number":
            assert n is not None
            semantic_features = torch.zeros((n, self.config.feature_dim), dtype=torch.float)
            anchor_features = torch.zeros((n, self.config.feature_dim), dtype=torch.float)

        elif mode == "tensors":
            assert tensors is not None and "anchor_features" in tensors
            semantic_features = tensors["semantic_features"]
            anchor_features = tensors["anchor_features"]

            self.gaussian_mlps["opacity"].load_state_dict(tensors["opacity_mlp"])
            self.gaussian_mlps["cov"].load_state_dict(tensors["cov_mlp"])
            self.gaussian_mlps["color"].load_state_dict(tensors["color_mlp"])
            if "feature_bank_mlp" in tensors:
                self.gaussian_mlps["feature_bank"].load_state_dict(tensors["feature_bank_mlp"])
                self.config.use_feature_bank = True

        else:
            raise ValueError(f"Unsupported mode {mode}")

        semantic_features = nn.Parameter(semantic_features, requires_grad=True)
        anchor_features = nn.Parameter(anchor_features, requires_grad=True)

        property_dict = {
            "semantic_features": semantic_features,
            "anchor_features": anchor_features,
        }
        return property_dict

    def training_setup_extra_properties(self, module, *args, **kwargs):
        [mlp_optimizer, constant_lr_optimizer], [mlp_scheduler] = super().training_setup_extra_properties(
            module, *args, **kwargs
        )

        optimization_config = self.config.optimization
        constant_lr_optimizer: torch.optim.Adam
        constant_lr_optimizer.add_param_group(
            {
                "params": self.gaussians["semantic_features"],
                "lr": optimization_config.semantic_features_lr,
                "name": "semantic_features",
            }
        )

        return [mlp_optimizer, constant_lr_optimizer], [mlp_scheduler]

    @property
    def get_semantic_features(self) -> torch.Tensor:
        return self.gaussians["semantic_features"]


@dataclass
class ImplicitGridOptimizationConfig(GridOptimizationConfig, RefinedScaffoldOptimizationConfigMixin):
    pass


@dataclass
class ImplicitGridGaussian(GridGaussian, ScaffoldGaussianMixin):
    optimization: ImplicitGridOptimizationConfig = field(default_factory=lambda: ImplicitGridOptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return ImplicitGridGaussianModel(self)


class ImplicitGridGaussianModel(RefinedScaffoldGaussianModelMixin, GridGaussianModel):
    config: ImplicitGridGaussian


@dataclass
class ImplicitLoDGridOptimizationConfig(LoDGridOptimizationConfig, RefinedScaffoldOptimizationConfigMixin):
    pass


@dataclass
class ImplicitLoDGridGaussian(LoDGridGaussian, ScaffoldGaussianMixin):
    optimization: ImplicitLoDGridOptimizationConfig = field(default_factory=lambda: ImplicitLoDGridOptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return ImplicitLoDGridGaussianModel(self)


class ImplicitLoDGridGaussianModel(RefinedScaffoldGaussianModelMixin, LoDGridGaussianModel):
    config: ImplicitLoDGridGaussian
