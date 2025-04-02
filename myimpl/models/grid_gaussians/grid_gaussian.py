from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import lightning
import torch
import torch.nn as nn

from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.network_factory import NetworkFactory

from .utils import GridGaussianUtils

__all__ = ["GridGaussian", "GridGaussianModel", "GridOptimizationConfig"]


@dataclass
class GridOptimizationConfig:
    anchors_lr_init: float = 0.0

    scales_lr: float = 0.007

    offsets_lr_init: float = 0.01

    offsets_lr_scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {
                "lr_final": 0.0001,
                "max_steps": 40_000,
            },
        }
    )

    spatial_lr_scale: float = -1

    sh_degree_up_interval: int = 1_000

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})


@dataclass
class GridGaussian(Gaussian):
    n_offsets: int = 10

    padding: float = 0.0
    """Padding for the octree grid"""

    base_layer: int = 11
    """The base layer of the octree"""

    default_voxel_size: float = -1.0

    view_dim: int = 3

    n_appearance_embedding_dims: int = 0

    color_mode: Literal["RGB", "SHs"] = "RGB"

    sh_degree: int = 3

    optimization: GridOptimizationConfig = field(default_factory=lambda: GridOptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return GridGaussianModel(self)


class GridGaussianModel(GaussianModel):
    _extra_property_names: List[str] = []
    _extra_buffer_names: List[str] = []
    """
    Parameters: means (anchors), offsets, scales
    Buffers: _voxel_size, _grid_origin
    """

    def __init__(self, config: GridGaussian):
        super().__init__()
        self.config = config

        names = ["means", "offsets", "scales"]
        self._names = tuple(names + self._extra_property_names)

        buffer_names = ["_voxel_size", "_grid_origin", "_activate_sh_degree"]
        self._buffer_names = tuple(buffer_names + self._extra_buffer_names)

    def before_setup_properties_from_pcd(
        self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs
    ):
        pass

    def setup_grid(self, points: torch.Tensor):
        voxel_size, grid_origin = GridGaussianUtils.build_grid(
            points, default_voxel_size=self.config.default_voxel_size
        )
        self.register_buffer("_voxel_size", voxel_size)
        self.register_buffer("_grid_origin", grid_origin)

    def setup_from_pcd(self, xyz, rgb, *args, **kwargs):
        points = torch.from_numpy(xyz).float()
        self.setup_grid(points)
        self.register_buffer("_activate_sh_degree", torch.tensor(0, dtype=torch.int))

        fused_point_cloud = GridGaussianUtils.voxelize(points, self.voxel_size, self.xyz2grid, self.grid2xyz)
        property_dict = self.get_init_properties(fused_point_cloud=fused_point_cloud, mode="pcd")
        self.before_setup_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        pass

    def setup_from_number(self, n, *args, **kwargs):
        self.register_buffer("_voxel_size", torch.tensor(0, dtype=torch.float))
        self.register_buffer("_grid_origin", torch.zeros((3,), dtype=torch.float))
        self.register_buffer("_activate_sh_degree", torch.tensor(0, dtype=torch.int))

        property_dict = self.get_init_properties(n=n, mode="number")
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def setup_from_tensors(self, tensors, *args, **kwargs):
        """
        setup from state_dict
        """
        assert all([n in tensors for n in self.get_buffer_names()])
        for n in self.get_buffer_names():
            self.register_buffer(n, tensors[n])

        property_dict = self.get_init_properties(tensors=tensors, mode="tensors")
        for name, value in property_dict.items():
            self.set_property(name, value)

    def training_setup(self, module: "lightning.LightningModule"):
        basic_optimizers, basic_schedulers = self.training_setup_basic_properties(module=module)
        lod_optimizers, lod_schedulers = self.training_setup_lod_properties(module=module)
        extra_optimizers, extra_schedulers = self.training_setup_extra_properties(module=module)
        return (
            basic_optimizers + lod_optimizers + extra_optimizers,
            basic_schedulers + lod_schedulers + extra_schedulers,
        )

    def get_init_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        *args,
        **kwargs,
    ):
        property_dict = {}
        property_dict.update(self.get_basic_properties(fused_point_cloud, n, tensors, mode, *args, **kwargs))
        property_dict.update(self.get_lod_properties(fused_point_cloud, n, tensors, mode, *args, **kwargs))
        property_dict.update(self.get_extra_properties(fused_point_cloud, n, tensors, mode, *args, **kwargs))
        return property_dict

    def get_basic_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        *args,
        **kwargs,
    ):
        if mode == "pcd":
            assert fused_point_cloud is not None
            from simple_knn._C import distCUDA2

            n_anchors, n_offsets = fused_point_cloud.shape[0], self.config.n_offsets
            anchors = fused_point_cloud
            offsets = anchors.new_zeros((n_anchors, n_offsets, 3))
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        elif mode == "number":
            assert n is not None
            anchors = torch.zeros((n, 3)).float()
            offsets = torch.zeros((n, self.config.n_offsets, 3)).float()
            scales = torch.zeros((n, 6)).float()
        elif mode == "tensors":
            assert tensors is not None and "means" in tensors
            anchors = tensors["means"]
            offsets = anchors.new_zeros((anchors.shape[0], self.config.n_offsets, 3))
            dist2 = torch.clamp_min(distCUDA2(anchors.cuda()), 0.0000001).to(anchors.device)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        anchors = nn.Parameter(anchors, requires_grad=True)
        offsets = nn.Parameter(offsets, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)

        property_dict = {
            "means": anchors,
            "scales": scales,
            "offsets": offsets,
        }
        return property_dict

    def get_lod_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        levels: Optional[torch.Tensor] = None,
        extra_levels: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """if model is LoD, add levels and extra_levels"""
        return {}

    def get_extra_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        *args,
        **kwargs,
    ):
        """
        for implicit model, add anchor_features and create mlps
        for explicit model, add extra properties including opacities, scales, rots and color.
        """
        return {}

    def training_setup_basic_properties(
        self,
        module: lightning.LightningModule,
        *args,
        **kwargs,
    ):
        spatial_lr_scale = self.config.optimization.spatial_lr_scale
        if spatial_lr_scale <= 0:
            spatial_lr_scale = module.trainer.datamodule.dataparser_outputs.camera_extent
        assert spatial_lr_scale > 0

        optimization_config = self.config.optimization
        optimizer_factory = optimization_config.optimizer
        offsets_lr_init = optimization_config.offsets_lr_init * spatial_lr_scale
        offsets_optimizer = optimizer_factory.instantiate(
            [{"params": [self.gaussians["offsets"]], "name": "offsets"}],
            lr=offsets_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(offsets_optimizer, module)
        optimization_config.offsets_lr_scheduler.lr_final *= spatial_lr_scale
        offsets_scheduler = optimization_config.offsets_lr_scheduler.instantiate().get_scheduler(
            offsets_optimizer,
            offsets_lr_init,
        )
        # constant properties
        l = [
            {"params": self.gaussians["means"], "lr": optimization_config.anchors_lr_init, "name": "means"},
            {"params": self.gaussians["scales"], "lr": optimization_config.scales_lr, "name": "scales"},
        ]
        constant_lr_optimizer = optimizer_factory.instantiate(l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(constant_lr_optimizer, module)

        module.on_train_batch_end_hooks.append(self.activate_sh_degree_update)

        return [offsets_optimizer, constant_lr_optimizer], [offsets_scheduler]

    def training_setup_lod_properties(self, module, *args, **kwargs):
        return [], []

    def training_setup_extra_properties(self, module, *args, **kwargs):
        return [], []

    def grid2xyz(self, grids: torch.Tensor, voxel_size: float):
        return GridGaussianUtils.grid_to_point(
            grids, voxel_size, grid_origin=self.grid_origin, padding=self.config.padding
        )

    def xyz2grid(self, points: torch.Tensor, voxel_size: float):
        return GridGaussianUtils.point_to_grid(
            points, voxel_size, grid_origin=self.grid_origin, padding=self.config.padding
        )

    def get_property_names(self):
        return self._names

    def get_buffer_names(self):
        return self._buffer_names

    @property
    def voxel_size(self) -> float:
        self._voxel_size: torch.Tensor
        return self._voxel_size.item()

    @property
    def grid_origin(self) -> torch.Tensor:
        self._grid_origin: torch.Tensor
        return self._grid_origin

    @property
    def n_anchors(self) -> int:
        return self.get_anchors.shape[0]

    @property
    def n_offsets(self) -> int:
        return self.config.n_offsets

    @property
    def n_primitives(self) -> int:
        return self.n_anchors * self.n_offsets

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.get_anchors

    @property
    def get_anchors(self) -> torch.Tensor:
        """[N_anchors, 3]"""
        return self.gaussians["means"]

    @property
    def get_scalings(self) -> torch.Tensor:
        """[N, 6], 3 for offsets, 3 for scaling"""
        return self.scale_activation(self.gaussians["scales"])

    @property
    def get_offsets(self) -> torch.Tensor:
        """[N_anchors, N_offsets, 3]"""
        return self.gaussians["offsets"]

    @property
    def color_dim(self):
        if self.config.color_mode == "RGB":
            return 3
        elif self.config.color_mode == "SHs":
            return 3 * (self.config.sh_degree + 1) ** 2
        else:
            raise ValueError(f"Unknown color mode: {self.config.color_mode}")

    @classmethod
    def activate_sh_degree_update(
        cls,
        output,
        batch,
        gaussian_model,
        global_step,
        pl_module: lightning.LightningModule,
    ):
        if gaussian_model.config.color_mode == "SHs":
            if (
                global_step % gaussian_model.config.optimization.sh_degree_up_interval != 0
                or gaussian_model.activate_sh_degree >= gaussian_model.config.sh_degree
            ):
                return
            gaussian_model.register_buffer("_activate_sh_degree", gaussian_model._activate_sh_degree + 1)

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
        if self.config.color_mode == "SHs":
            if getattr(self, "_activate_sh_degree", None) is None:
                self.register_buffers("_activate_sh_degree", torch.tensor(0))
            return self._activate_sh_degree.item()
        return 0

    def _add_optimizer_after_backward_hook_if_available(self, optimizer, pl_module):
        hook = getattr(optimizer, "on_after_backward", None)
        if hook is None:
            return
        pl_module.on_after_backward_hooks.append(hook)

    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales)

    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.log(scales)

    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(rotations)

    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return rotations
