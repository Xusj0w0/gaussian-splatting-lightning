from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from gsplat.cuda._wrapper import spherical_harmonics
from pytorch3d.transforms import quaternion_multiply

from internal.cameras.cameras import Camera
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import inverse_sigmoid
from internal.utils.network_factory import NetworkFactory

from .base import GridGaussianModelBase
from .utils import init_weight

__all__ = ["ScaffoldGaussianMixin", "ScaffoldGaussianModelMixin", "ScaffoldOptimizationConfigMixin"]


@dataclass
class ScaffoldOptimizationConfigMixin:
    anchor_features_lr: float = 0.0075

    opacity_mlp_lr_init: float = 0.002
    opacity_mlp_lr_final: float = 0.00002

    cov_mlp_lr_init: float = 0.004
    cov_mlp_lr_final: float = 0.004

    color_mlp_lr_init: float = 0.008
    color_mlp_lr_final: float = 0.00005

    feature_bank_mlp_lr_init: float = 0.01
    feature_bank_mlp_lr_final: float = 0.00001

    mlp_optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})

    mlp_scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {"max_steps": None},
        }
    )


@dataclass
class ScaffoldGaussianMixin:
    feature_dim: int = 32

    mlp_n_layers: int = 2

    use_feature_bank: bool = False

    tcnn: bool = False


class ScaffoldGaussianModelMixin:  # GridGaussianModel,
    config: ScaffoldGaussianMixin
    _extra_property_names: List[str] = ["anchor_features"]

    def calculate_implicit_properties(
        self: Union["ScaffoldGaussianModelMixin", GridGaussianModelBase],
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
        features = self.get_anchor_features[anchor_mask]
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

    def create_mlps(self):
        self.gaussian_mlps = nn.ModuleDict()
        # create mlps
        # opacity: return 1*n_offsets
        self.gaussian_mlps["opacity"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
            n_input_dims=self.config.feature_dim,  # try: remove viewdirs + self.config.view_dim,
            n_output_dims=self.n_offsets,
            n_layers=self.config.mlp_n_layers,
            n_neurons=self.config.feature_dim,
            activation="ReLU",
            output_activation="Tanh",
        )
        # cov: return 7*n_offsets
        # 3 for scales, multiply with anchor-level scales to get gaussian scales
        # 4 for rotations
        self.gaussian_mlps["cov"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
            n_input_dims=self.config.feature_dim + self.config.view_dim,
            n_output_dims=7 * self.n_offsets,
            n_layers=self.config.mlp_n_layers,
            n_neurons=self.config.feature_dim,
            activation="ReLU",
            output_activation="None",
        )
        # color: return 3*n_offsets
        self.gaussian_mlps["color"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
            n_input_dims=self.config.feature_dim + self.config.view_dim + self.config.n_appearance_embedding_dims,
            n_output_dims=self.color_dim * self.n_offsets,
            n_layers=self.config.mlp_n_layers,
            n_neurons=self.config.feature_dim,
            activation="ReLU",
            output_activation="Sigmoid",
        )
        if self.config.use_feature_bank:
            self.gaussian_mlps["feature_bank"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
                n_input_dims=self.config.view_dim,
                n_output_dims=3,
                n_layers=2,
                n_neurons=self.config.feature_dim,
                activation="ReLU",
                output_activation="None",
            )

        # for mlp in self.gaussian_mlps.values():
        #     mlp.apply(init_weight)

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
            anchor_features = torch.zeros((n_anchors, self.config.feature_dim), dtype=torch.float)
            # anchor_features = torch.normal(0.0, 0.02, (n_anchors, self.config.feature_dim), dtype=torch.float)

        elif mode == "number":
            assert n is not None
            anchor_features = torch.zeros((n, self.config.feature_dim), dtype=torch.float)

        elif mode == "tensors":
            assert tensors is not None and "anchor_features" in tensors
            anchor_features = tensors["anchor_features"]

            self.gaussian_mlps["opacity"].load_state_dict(tensors["opacity_mlp"])
            self.gaussian_mlps["cov"].load_state_dict(tensors["cov_mlp"])
            self.gaussian_mlps["color"].load_state_dict(tensors["color_mlp"])
            if "feature_bank_mlp" in tensors:
                self.gaussian_mlps["feature_bank"].load_state_dict(tensors["feature_bank_mlp"])
                self.config.use_feature_bank = True

        else:
            raise ValueError(f"Unsupported mode {mode}")

        anchor_features = nn.Parameter(anchor_features, requires_grad=True)

        property_dict = {
            "anchor_features": anchor_features,
        }
        return property_dict

    def training_setup_extra_properties(self, module, *args, **kwargs):
        if self.config.optimization.mlp_scheduler.max_steps is None:
            self.config.optimization.mlp_scheduler.max_steps = module.trainer.max_steps

        optimization_config = self.config.optimization
        optimizer_factory = self.config.optimization.optimizer
        mlp_optimizer_factory = self.config.optimization.mlp_optimizer
        mlp_scheduler_factory = self.config.optimization.mlp_scheduler

        # constant properties
        # fmt: off
        l = [
            {"params": self.gaussians["anchor_features"], "lr": optimization_config.anchor_features_lr, "name": "anchor_features"},
        ]
        # fmt: on
        constant_lr_optimizer = optimizer_factory.instantiate(l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(constant_lr_optimizer, module)

        mlp_l = [
            {
                "params": self.gaussian_mlps["opacity"].parameters(),
                "lr": optimization_config.opacity_mlp_lr_init,
                "name": "opacity_mlp",
            },
            {
                "params": self.gaussian_mlps["cov"].parameters(),
                "lr": optimization_config.cov_mlp_lr_init,
                "name": "cov_mlp",
            },
            {
                "params": self.gaussian_mlps["color"].parameters(),
                "lr": optimization_config.color_mlp_lr_init,
                "name": "color_mlp",
            },
        ]
        if self.config.use_feature_bank:
            mlp_l.append(
                {
                    "params": self.gaussian_mlps["feature_bank"].parameters(),
                    "lr": optimization_config.feature_bank_mlp_lr_init,
                    "name": "feature_bank_mlp",
                }
            )
        mlp_optimizer = mlp_optimizer_factory.instantiate(mlp_l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(mlp_optimizer, module)
        scheduler_lr_finals = [
            optimization_config.opacity_mlp_lr_final,
            optimization_config.cov_mlp_lr_final,
            optimization_config.color_mlp_lr_final,
        ]
        if self.config.use_feature_bank:
            scheduler_lr_finals.append(optimization_config.feature_bank_mlp_lr_final)
        mlp_scheduler = mlp_scheduler_factory.instantiate().get_schedulers(mlp_optimizer, scheduler_lr_finals)

        return [mlp_optimizer, constant_lr_optimizer], [mlp_scheduler]

    def train(self, mode=True):
        for mlp in self.gaussian_mlps.values():
            mlp.train()
        return super().train(mode)

    def eval(self):
        for mlp in self.gaussian_mlps.values():
            mlp.eval()
        return super().eval()

    @property
    def get_anchor_features(self):
        """[N, C]"""
        return self.gaussians["anchor_features"]

    @property
    def get_features(self):
        """save_gaussians() will call `get_features`"""
        return self.get_anchor_features.new_zeros((self.n_anchors, 1, 3))

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
