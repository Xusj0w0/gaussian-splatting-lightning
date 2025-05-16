from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import lightning
import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from internal.cameras.cameras import Camera
from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.renderers.gsplat_v1_renderer import GSplatV1
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import inverse_sigmoid
from internal.utils.network_factory import NetworkFactory

from .utils import GridFactory


@dataclass
class GridOptimizationConfigBase:
    means_lr_init: float = 0.0

    scales_lr: float = 0.007

    rotations_lr_init: float = 0.001

    offsets_lr_init: float = 0.01

    offsets_lr_scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {
                "lr_final": 0.0001,
                "max_steps": None,
            },
        }
    )

    spatial_lr_scale: float = -1

    sh_degree_up_interval: int = 1_000

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})


@dataclass
class GridGaussianBase(Gaussian):
    n_offsets: int = 10

    padding: float = 0.0
    """Padding for the octree grid"""

    default_voxel_size: float = -1.0

    outlier_ratio: float = 0.001
    """Outlier ratio of origin point cloud"""

    extend_ratio: float = 0.1
    """Extend ratio of the grid relative to original point cloud"""

    view_dim: int = field(default=3, init=False)

    view_sh_level: int = 0
    """
    SH level for view-dependent encoding:
    - 0: no sh encoding, [B, 3]
    - D: enable sh encoding, [B, D^2]. **D <= 4**
    """

    n_appearance_embedding_dims: int = 0

    color_mode: Literal["RGB", "SHs"] = "RGB"

    sh_degree: int = 3

    optimization: GridOptimizationConfigBase = field(default_factory=lambda: GridOptimizationConfigBase())

    def __post_init__(self):
        if self.view_sh_level > 4:
            raise ValueError(f"SH level {self.view_sh_level} is not supported, max is 4")
        if self.view_sh_level == 0:
            self.view_dim = 3
        else:
            self.view_dim = self.view_sh_level**2


class GridGaussianModelBase(GaussianModel):
    _extra_property_names: List[str] = []
    _extra_buffer_names: List[str] = []
    """
    Parameters: means (anchors), offsets, scales
    Buffers: _voxel_size, _transforms
    """

    def __init__(self, config: GridGaussianBase):
        super().__init__()
        self.config = config

        names = ["means", "offsets", "scales", "rotations"]
        self._names = tuple(names + self._extra_property_names)

        buffer_names = ["_voxel_size", "_transforms", "_grid_bbox", "_activate_sh_degree"]
        self._buffer_names = tuple(buffer_names + self._extra_buffer_names)

        if self.config.color_mode == "SHs":
            self.register_buffer("_activate_sh_degree", torch.tensor(0, dtype=torch.int))

    @torch.no_grad()
    def filter_anchor_by_preprojection(
        self, viewpoint_camera: Camera, anchor_mask: Optional[torch.Tensor] = None, **kwargs
    ):
        if anchor_mask is None:
            anchor_mask = self.get_anchors.new_ones((self.get_anchors.shape[0],), dtype=torch.bool)
        means = self.get_anchors[anchor_mask]
        scales = self.get_scalings[anchor_mask][:, :3]
        quats = means.new_zeros((means.shape[0], 4))
        quats[:, 0] = 1.0

        processed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
        radii = GSplatV1.project(
            processed_camera,
            means3d=means,
            scales=scales,
            quats=quats,
            anti_aliased=False,
        )[0]

        _anchor_mask = anchor_mask.clone()
        _anchor_mask[anchor_mask] = radii.squeeze(0) > 0
        return _anchor_mask

    def before_setup_properties_from_pcd(
        self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs
    ):
        pass

    def setup_grid(self, points: torch.Tensor, transforms: str = ""):
        voxel_size, transforms, grid_bbox = GridFactory.build_grid(
            points,
            default_voxel_size=self.config.default_voxel_size,
            transforms=transforms,
            outlier_ratio=self.config.outlier_ratio,
            extend_ratio=self.config.extend_ratio,
        )
        self.register_buffer("_voxel_size", voxel_size)
        self.register_buffer("_transforms", transforms)
        self.register_buffer("_grid_bbox", grid_bbox)

    def setup_from_pcd(self, xyz, rgb, transforms, *args, **kwargs):
        points = torch.from_numpy(xyz).float()
        self.setup_grid(points, transforms)

        fused_point_cloud = GridFactory.voxelize(points, self.voxel_size, self.xyz2grid, self.grid2xyz)
        property_dict = self.get_init_properties(fused_point_cloud=fused_point_cloud, mode="pcd", *args, **kwargs)
        self.before_setup_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        pass

    def setup_from_number(self, n, *args, **kwargs):
        self.register_buffer("_voxel_size", torch.tensor(0, dtype=torch.float))
        transforms = torch.zeros((7,), dtype=torch.float)
        transforms[0] = 1.0
        self.register_buffer("_transforms", transforms)
        self.register_buffer("_grid_bbox", torch.zeros((6,), dtype=torch.float))

        property_dict = self.get_init_properties(n=n, mode="number", *args, **kwargs)
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def setup_from_tensors(self, tensors, *args, **kwargs):
        """
        setup from state_dict
        """
        buffer_names = self.get_buffer_names()
        for n in tensors:
            if n in buffer_names:
                self.register_buffer(n, tensors[n])

        property_dict = self.get_init_properties(tensors=tensors, mode="tensors", *args, **kwargs)
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
            rotations = anchors.new_zeros((n_anchors, 4))
            rotations[:, 0] = 1.0

        elif mode == "number":
            assert n is not None
            anchors = torch.zeros((n, 3)).float()
            offsets = torch.zeros((n, self.config.n_offsets, 3)).float()
            scales = torch.zeros((n, 6)).float()
            rotations = anchors.new_zeros((n, 4))
            rotations[:, 0] = 1.0

        elif mode == "tensors":
            assert tensors is not None and "anchors" in tensors
            anchors = tensors["anchors"]
            offsets = tensors["offsets"]
            scales = tensors["scales"]
            rotations = tensors["rotations"]

        else:
            raise ValueError(f"Unsupported mode {mode}")

        anchors = nn.Parameter(anchors, requires_grad=True)
        offsets = nn.Parameter(offsets, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)
        rotations = nn.Parameter(rotations, requires_grad=True)

        property_dict = {
            "means": anchors,
            "scales": scales,
            "offsets": offsets,
            "rotations": rotations,
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
        if self.config.optimization.offsets_lr_scheduler.max_steps is None:
            self.config.optimization.offsets_lr_scheduler.max_steps = module.trainer.max_steps

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
            {"params": self.gaussians["means"], "lr": optimization_config.means_lr_init, "name": "means"},
            {"params": self.gaussians["scales"], "lr": optimization_config.scales_lr, "name": "scales"},
            {"params": self.gaussians["rotations"], "lr": optimization_config.rotations_lr_init, "name": "rotations"},
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
        return GridFactory.grid_to_point(grids, voxel_size, transforms=self.transforms, padding=self.config.padding)

    def xyz2grid(self, points: torch.Tensor, voxel_size: float):
        return GridFactory.point_to_grid(points, voxel_size, transforms=self.transforms, padding=self.config.padding)

    def normalize_xyz(self, points: torch.Tensor):
        transforms = self.transforms.to(points.device)
        transformed = points @ transforms[:3, :3].T + transforms[:3, 3]
        bbox_min, bbox_max = self.grid_bbox
        bbox_min, bbox_max = bbox_min.to(points.device), bbox_max.to(points.device)
        return (transformed - bbox_min) / (bbox_max - bbox_min)

    def get_property_names(self):
        return self._names

    def get_buffer_names(self):
        return self._buffer_names

    @property
    def voxel_size(self) -> float:
        self._voxel_size: torch.Tensor
        return self._voxel_size.item()

    @property
    def transforms(self) -> torch.Tensor:
        """transposed transform matrix"""
        self._transform_matrix: torch.Tensor
        if not hasattr(self, "_transform_matrix"):
            self._transform_matrix = torch.eye(4).to(self._transforms)
            self._transform_matrix[:3, :3] = quaternion_to_matrix(self._transforms[:4])
            self._transform_matrix[:3, 3] = self._transforms[4:]
        return self._transform_matrix

    @property
    def grid_bbox(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(min, max), ([3,], [3,])"""
        self._grid_bbox: torch.Tensor
        return (self._grid_bbox[:3], self._grid_bbox[3:])

    @property
    def n_anchors(self) -> int:
        return self.get_anchors.shape[0]

    @property
    def n_offsets(self) -> int:
        return self.config.n_offsets

    @property
    def n_primitives(self) -> int:
        return self.n_anchors * self.n_offsets

    def get_n_gaussians(self):
        return self.n_anchors

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.get_anchors

    def get_means(self) -> torch.Tensor:
        return self.gaussians["means"]

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
    def get_rotations(self) -> torch.Tensor:
        return self.rotation_activation(self.gaussians["rotations"])

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

    def pre_activate_all_properties(self):
        pass

    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales)

    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.log(scales)

    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(rotations)

    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return rotations

    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(opacities)

    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(opacities)
