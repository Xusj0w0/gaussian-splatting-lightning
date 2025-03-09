import math
import sched
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import lightning
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from internal.cameras.cameras import Camera, Cameras
from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import (build_scaling_rotation,
                                          inverse_sigmoid, strip_symmetric)
from internal.utils.network_factory import NetworkFactory
from myimpl.models.octree_gaussian import OctreeGaussian, OctreeGaussianModel
from myimpl.models.octree_gaussian import \
    OpimizationConfig as OctreeOpimizationConfig
from myimpl.utils.octree_utils import init_weight, knn


@dataclass
class OpimizationConfig(OctreeOpimizationConfig):
    sh_degree_up_interval: int = 1_000

    anchor_features_lr: float = 0.0075

    rotation_lr: float = 0.002

    opacity_mlp_lr_init: float = 0.002
    opacity_mlp_lr_final: float = 0.00002

    cov_mlp_lr_init: float = 0.004
    cov_mlp_lr_final: float = 0.004

    color_mlp_lr_init: float = 0.008
    color_mlp_lr_final: float = 0.00005

    feature_bank_mlp_lr_init: float = 0.01
    feature_bank_mlp_lr_final: float = 0.00001

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})

    mlp_scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {"max_steps": 100_000},
        }
    )


@dataclass
class ScaffoldLoDGaussian(OctreeGaussian):
    feature_dim: int = 32

    view_dim: int = 3

    use_feature_bank: bool = False

    n_appearance_embedding_dims: int = 0

    tcnn: bool = False

    color_mode: Literal["RGB", "SHs"] = "RGB"

    sh_degree: int = 3

    optimization: OpimizationConfig = field(default_factory=lambda: OpimizationConfig())

    def instantiate(self, *args, **kwargs) -> "ScaffoldLoDGaussianModel":
        return ScaffoldLoDGaussianModel(self)


class ScaffoldLoDGaussianModel(OctreeGaussianModel):
    def __init__(self, config: ScaffoldLoDGaussian) -> None:
        super().__init__(config)
        self.config = config

        names = [
            "rotations",  # [N, 4]
            "anchor_features",
        ] + self.get_extra_property_names()
        self._names = tuple(list(self._names) + names)

        self._buffer_names = tuple(list(self._buffer_names) + ["_activate_sh_degree"])
        self.register_buffer("_activate_sh_degree", torch.tensor(0, dtype=torch.int))

        self.gaussian_mlps = nn.ModuleDict()

    def create_mlps(self):
        # create mlps
        # opacity: return 1*n_offsets
        self.gaussian_mlps["opacity"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
            n_input_dims=self.config.feature_dim + self.config.view_dim,
            n_output_dims=self.config.n_offsets,
            n_layers=2,
            n_neurons=self.config.feature_dim,
            activation="ReLU",
            output_activation="Tanh",
        )
        # cov: return 7*n_offsets
        # 3 for scales, multiply with anchor-level scales to get gaussian scales
        # 4 for rotations
        self.gaussian_mlps["cov"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
            n_input_dims=self.config.feature_dim + self.config.view_dim,
            n_output_dims=7 * self.config.n_offsets,
            n_layers=2,
            n_neurons=self.config.feature_dim,
            activation="ReLU",
            output_activation="None",
        )
        # color: return 3*n_offsets
        self.gaussian_mlps["color"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
            n_input_dims=self.config.feature_dim + self.config.view_dim + self.config.n_appearance_embedding_dims,
            n_output_dims=self.color_dim * self.config.n_offsets,
            n_layers=2,
            n_neurons=self.config.feature_dim,
            activation="ReLU",
            output_activation="None",
        )
        if self.config.use_feature_bank > 0:
            self.gaussian_mlps["feature_bank"] = NetworkFactory(tcnn=self.config.tcnn).get_network(
                n_input_dims=self.config.view_dim,
                n_output_dims=3,
                n_layers=2,
                n_neurons=self.config.feature_dim,
                activation="ReLU",
                output_activation="None",
            )

    def train(self, mode=True):
        for mlp in self.gaussian_mlps.values():
            mlp.train(mode)
        return super().train(mode)

    def eval(self):
        for mlp in self.gaussian_mlps.values():
            mlp.eval()
        return super().eval()

    def get_extra_property_names(self):
        return []

    def before_setup_set_properties_from_pcd(
        self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs
    ):
        pass

    def setup_from_pcd(self, xyz, rgb, cameras, *args, **kwargs):
        super().setup_from_pcd(xyz=xyz, rgb=rgb, cameras=cameras, *args, **kwargs)

        n_anchors, n_offsets = self.get_anchors.shape[0], self.config.n_offsets

        rots = torch.zeros((n_anchors, 4), dtype=torch.float)
        rots[:, 0] = 1
        anchor_features = torch.zeros((n_anchors, self.config.feature_dim), dtype=torch.float)
        rotations = nn.Parameter(rots, requires_grad=False)
        anchor_features = nn.Parameter(anchor_features, requires_grad=True)

        property_dict = {
            "rotations": rotations,
            "anchor_features": anchor_features,
        }
        self.before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        self.update_properties(property_dict, strict=False)

        self.create_mlps()

    def before_setup_set_properties_from_number(self, n, property_dict, *args, **kwargs):
        pass

    def setup_from_number(self, n, *args, **kwargs):
        super().setup_from_number(n, *args, **kwargs)

        rots = torch.zeros((n, 4), dtype=torch.float)
        rots[:, 0] = 1
        anchor_features = torch.zeros((n, self.config.feature_dim), dtype=torch.float)

        rotations = nn.Parameter(rots.requires_grad_(True))
        anchor_features = nn.Parameter(anchor_features.requires_grad_(True))
        property_dict = {
            "rotations": rotations,
            "anchor_features": anchor_features,
        }
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        self.update_properties(property_dict, strict=False)

        self.create_mlps()

    def setup_from_tensors(self, tensors, *args, **kwargs):
        # TODO
        pass

    def training_setup(self, module: "lightning.LightningModule"):
        (offsets_optimizer, constant_lr_optimizer), (offsets_scheduler,) = super().training_setup(module)

        optimization_config = self.config.optimization
        optimizer_factory = self.config.optimization.optimizer

        # constant properties
        l = [
            {"params": self.gaussians["rotations"], "lr": optimization_config.rotation_lr, "name": "rotations"},
            {"params": self.gaussians["anchor_features"], "lr": optimization_config.anchor_features_lr, "name": "anchor_features",},  # fmt: skip
        ]
        for param_group in l:
            constant_lr_optimizer.add_param_group(param_group)

        # mlps
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

        mlp_optimizer = optimizer_factory.instantiate(mlp_l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(mlp_optimizer, module)
        mlp_scheduler_factory = self.config.optimization.mlp_scheduler
        scheduler_lr_finals = [
            optimization_config.opacity_mlp_lr_final,
            optimization_config.cov_mlp_lr_final,
            optimization_config.color_mlp_lr_final,
        ]
        if self.config.use_feature_bank:
            scheduler_lr_finals.append(optimization_config.feature_bank_mlp_lr_final)
        mlp_scheduler = mlp_scheduler_factory.instantiate().get_schedulers(mlp_optimizer, scheduler_lr_finals)

        optimizers = [offsets_optimizer, constant_lr_optimizer, mlp_optimizer]
        schedulers = [offsets_scheduler, mlp_scheduler]

        return optimizers, schedulers

    def on_train_batch_end(self, step, module):
        super().on_train_batch_end(step, module)

        if self.config.color_mode == "SHs":
            if (
                step % self.config.optimization.sh_degree_up_interval != 0
                or self.activate_sh_degree >= self.config.sh_degree
            ):
                return
            self.register_buffer("_activate_sh_degree", self._activate_sh_degree + 1)

    def get_property_names(self) -> Tuple[str, ...]:
        return self._names

    @staticmethod
    def _return_as_is(v):
        return v

    @property
    def get_rotations(self):
        """[N, 4]"""
        return self.rotation_activation(self.gaussians["rotations"])

    @property
    def get_anchor_features(self):
        """[N, C]"""
        return self.gaussians["anchor_features"]

    @property
    def get_features(self):
        """save_gaussians() will call `get_features`"""
        return self.get_xyz.new_zeros((self.n_anchors, 1, 3))

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

    @property
    def max_sh_degree(self):
        if self.config.color_mode == "RGB":
            return 0
        elif self.config.color_mode == "SHs":
            return self.config.sh_degree
        else:
            raise ValueError(f"Unknown color mode: {self.config.color_mode}")

    @property
    def activate_sh_degree(self):
        self._activate_sh_degree: torch.Tensor
        return self._activate_sh_degree.item()

    @property
    def color_dim(self):
        if self.config.color_mode == "RGB":
            return 3
        elif self.config.color_mode == "SHs":
            return 3 * (self.config.sh_degree + 1) ** 2
        else:
            raise ValueError(f"Unknown color mode: {self.config.color_mode}")
