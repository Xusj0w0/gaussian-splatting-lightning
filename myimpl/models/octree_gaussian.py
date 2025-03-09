import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import lightning
import numpy as np
import torch
import torch.nn as nn

from internal.cameras.cameras import Camera, Cameras
from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import (build_scaling_rotation,
                                          inverse_sigmoid, strip_symmetric)
from myimpl.utils.octree_utils import OctreeUtils, knn


@dataclass
class OpimizationConfig:
    progressive: bool = True
    """whether optimize anchor from coarse to fine progressively"""

    coarse_factor: float = 1.5

    coarse_iter: int = 10_000

    spatial_lr_scale: float = -1

    anchors_lr_init: float = 0.0
    # anchors_lr_scheduler: Scheduler = field(
    #     default_factory=lambda: {
    #         "class_path": "ExponentialDecayScheduler",
    #         "init_args": {
    #             "lr_final": 0.0,
    #             "max_steps": 40_000,
    #         },
    #     }
    # )

    offsets_lr_init: float = 0.01
    offsets_lr_scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {
                "lr_final": 0.0001,
                "max_steps": 100_000,
            },
        }
    )

    scales_lr: float = 0.007

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})


@dataclass
class OctreeGaussian(Gaussian):
    n_offsets: int = 10

    padding: float = 0.0
    """Padding for the octree grid"""

    fork: int = 2

    dist2level: Literal["floor", "round", "progressive"] = "floor"

    base_layer: int = 11
    """The base layer of the octree"""

    default_voxel_size: float = 0.02

    extend_ratio: float = 0.1
    """Extend ratio of the octree grid relative to original point cloud"""

    dist_ratio: float = 0.001
    """Filter distances between camera centers and points within (0, dist_ratio) and (1-dist_ratio, 1)"""

    max_level: int = -1

    optimize_from_level: int = -1

    visibility_threshold: float = 0.01

    optimization: OpimizationConfig = field(default_factory=lambda: OpimizationConfig())

    def instantiate(self, *args, **kwargs):
        return OctreeGaussianModel(self)


class OctreeGaussianModel(GaussianModel):
    """
    buffers: levels, init_level, standard_dist, voxel_size, visibility_threshold
    """

    def __init__(self, config: OctreeGaussian):
        super().__init__()
        self.config = config

        names = [
            "means",  # [N, 3], position of anchors
            "offsets",  # [N, N_offsets, 3]
            "scales",  # [N, 6], anchor-level scales, 3 for gaussian offsets, 3 for gaussian scales
            "levels",  # [N,]
            "extra_levels",  # [N,]
        ]
        self._names = tuple(names)

        buffer_names = [
            "_camera_infos",
            "_max_level",
            "_optimize_from_level",
            "_standard_dist",
            "_voxel_size",
            "_grid_origin",
            "_visibility_threshold",
        ]
        self._buffer_names = tuple(buffer_names)

    def get_property_names(self):
        return self._names

    def get_buffer_names(self):
        return self._buffer_names

    def on_train_batch_end(self, step: int, module: "lightning.LightningModule"):
        super().on_train_batch_end(step, module)

        self._activate_level = self.max_level
        if self.config.optimization.progressive:
            self._activate_level = np.searchsorted(self.coarse_intervals, step) + 1 + self.optimize_from_level

    def set_octree_properties(self, points: torch.Tensor, cameras: Cameras, *args, **kwargs):
        cam_centers = cameras.camera_center
        camera_infos = torch.cat([cam_centers, cam_centers.new_ones((cam_centers.shape[0], 1))], dim=-1)
        self._camera_infos: torch.Tensor
        self.register_buffer("_camera_infos", camera_infos)

        # calculate levels and register
        standard_dist, max_level = OctreeUtils.get_levels_by_distances(
            points, self.camera_infos, self.config.dist_ratio, self.config.fork
        )
        max_level = torch.tensor(self.config.max_level) if self.config.max_level > 0 else max_level
        optimize_from_level = (
            torch.tensor(self.config.optimize_from_level)
            if self.config.optimize_from_level > 0
            else (max_level / 2).int()
        )
        self._max_level: torch.Tensor; self._optimize_from_level: torch.Tensor; self._standard_dist: torch.Tensor  # fmt: skip
        self.register_buffer("_max_level", max_level)
        self.register_buffer("_optimize_from_level", optimize_from_level)
        self.register_buffer("_standard_dist", standard_dist)

        # calculate voxel grid and register
        voxel_size, grid_origin = OctreeUtils.setup_octree_voxel_grid(
            points,
            extend_ratio=self.config.extend_ratio,
            base_layer=self.config.base_layer,
            fork=self.config.fork,
            default_voxel_size=self.config.default_voxel_size,
            max_level=self.max_level,
        )
        self._voxel_size: torch.Tensor; self._grid_origin: torch.Tensor  # fmt: skip
        self.register_buffer("_voxel_size", voxel_size)
        self.register_buffer("_grid_origin", grid_origin)

        # get visibility_threshold
        vis_thresh = torch.tensor(self.config.visibility_threshold)
        if vis_thresh < 0.0:
            positions, levels = OctreeUtils.octree_sample(
                points,
                voxel_size=self.voxel_size,
                max_level=self.max_level,
                xyz2grid=self.xyz2grid,
                grid2xyz=self.grid2xyz,
                fork=self.config.fork,
            )
            mask = self.weed_out(positions, levels, 0.0)
            vis_thresh = torch.mean(mask.float())
        self._visibility_threshold: torch.Tensor
        self.register_buffer("_visibility_threshold", vis_thresh)

    def before_setup_set_properties_from_pcd(
        self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs
    ):
        pass

    def setup_from_pcd(self, xyz, rgb, cameras: Cameras, *args, **kwargs):
        points = torch.from_numpy(xyz).to(cameras[0].device).float()
        self.set_octree_properties(points, cameras)

        positions, levels = OctreeUtils.octree_sample(
            points,
            voxel_size=self.voxel_size,
            max_level=self.max_level,
            xyz2grid=self.xyz2grid,
            grid2xyz=self.grid2xyz,
            fork=self.config.fork,
        )
        weed_mask = self.weed_out(positions, levels, self.visibility_threshold)
        fused_point_cloud, levels = positions[weed_mask], levels[weed_mask]

        n_anchors, n_offsets = fused_point_cloud.shape[0], self.config.n_offsets

        offsets = fused_point_cloud.new_zeros((n_anchors, n_offsets, 3))
        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)
        scales = self.scale_inverse_activation((torch.sqrt(dist2))[..., None].repeat(1, 6))

        anchors = nn.Parameter(fused_point_cloud, requires_grad=True)
        offsets = nn.Parameter(offsets, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)
        levels = nn.Parameter(levels, requires_grad=False)
        extra_levels = nn.Parameter(levels.new_zeros((n_anchors,), dtype=torch.float), requires_grad=False)

        property_dict = {
            "means": anchors,
            "scales": scales,
            "offsets": offsets,
            "levels": levels,
            "extra_levels": extra_levels,
        }
        self.before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        pass

    def setup_from_number(self, n, *args, **kwargs):
        self.register_buffer("_camera_infos", torch.zeros((0, 4), dtype=torch.float))
        self.register_buffer("_max_level", torch.tensor(0, dtype=torch.int))
        self.register_buffer("_optimize_from_level", torch.tensor(0, dtype=torch.int))
        self.register_buffer("_standard_dist", torch.tensor(0, dtype=torch.float))
        self.register_buffer("_voxel_size", torch.tensor(0, dtype=torch.float))
        self.register_buffer("_grid_origin", torch.zeros((3,), dtype=torch.float))
        self.register_buffer("_visibility_threshold", torch.tensor(0, dtype=torch.float))

        anchors = torch.zeros((n, 3)).float()
        offsets = torch.zeros((n, self.config.n_offsets, 3)).float()
        scales = torch.zeros((n, 6)).float()
        levels = torch.zeros((n,)).int()
        extra_levels = torch.zeros((n,)).float()

        anchors = nn.Parameter(anchors, requires_grad=True)
        offsets = nn.Parameter(offsets, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)
        levels = nn.Parameter(levels, requires_grad=False)
        extra_levels = nn.Parameter(extra_levels, requires_grad=False)

        property_dict = {
            "means": anchors,
            "scales": scales,
            "offsets": offsets,
            "levels": levels,
            "extra_levels": extra_levels,
        }
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def setup_from_tensors(self, tensors, *args, **kwargs):
        pass

    def training_setup(self, module: "lightning.LightningModule"):
        spatial_lr_scale = self.config.optimization.spatial_lr_scale
        if spatial_lr_scale <= 0:
            spatial_lr_scale = module.trainer.datamodule.dataparser_outputs.camera_extent
        assert spatial_lr_scale > 0

        optimization_config = self.config.optimization

        optimizer_factory = self.config.optimization.optimizer

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

        return [offsets_optimizer, constant_lr_optimizer], [offsets_scheduler]

    def predict_level(self, dists: torch.Tensor):
        return OctreeUtils.predict_level(dists, standard_dist=self.standard_dist, fork=self.config.fork)

    def map_to_int_level(self, pred_level: torch.Tensor, cur_level: int):
        return OctreeUtils.map_to_int_level(pred_level, cur_level, dist2level=self.config.dist2level)

    def grid2xyz(self, grids: torch.Tensor, voxel_size: float):
        return OctreeUtils.grid_to_point(grids, voxel_size, grid_origin=self.grid_origin, padding=self.config.padding)

    def xyz2grid(self, points: torch.Tensor, voxel_size: float):
        return OctreeUtils.point_to_grid(points, voxel_size, grid_origin=self.grid_origin, padding=self.config.padding)

    def weed_out(self, anchors: torch.Tensor, levels: torch.Tensor, vis_thresh: float, use_chunk: bool = True):
        return OctreeUtils.weed_out_mask_by_level(
            anchors,
            levels,
            vis_thresh,
            cam_infos=self.camera_infos,
            predict_level_fn=self.predict_level,
            int_level_fn=lambda x: self.map_to_int_level(x, self.max_level)[0],
            use_chunk=use_chunk,
        )

    def mask_anchors_by_camera(self, viewpoint_camera: Camera):
        dists = torch.sqer(torch.sum((self.get_anchors - viewpoint_camera.camera_center) ** 2, dim=-1))
        pred_level = self.predict_level(dists) + self.get_extra_levels
        int_level_pkg = self.map_to_int_level(pred_level, self.activate_level)
        anchor_mask = self.get_levels <= int_level_pkg[0]

        return anchor_mask, int_level_pkg

    def generate_gaussian_primitives(
        self, viewpoint_camera: Camera, int_level_pkg: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ):
        """
        `int_level_pkg` is a tuple, the second is frac part of pred level (if dist2level == "progressive"),
        used for smoothing cross level artifacts.
        """
        pass

    @property
    def n_anchors(self):
        return self.get_anchors.shape[0]

    @property
    def n_offsets(self):
        return self.config.n_offsets

    @property
    def n_primitives(self):
        return self.n_anchors * self.n_offsets

    @property
    def get_xyz(self):
        return self.get_anchors

    @property
    def get_anchors(self):
        """[N_anchors, 3]"""
        return self.gaussians["means"]

    @property
    def get_scalings(self):
        """[N, 6], 3 for offsets, 3 for scaling"""
        return self.scale_activation(self.gaussians["scales"])

    @property
    def get_offsets(self):
        """[N_anchors, N_offsets, 3]"""
        return self.gaussians["offsets"]

    @property
    def get_levels(self):
        """[N_anchors, 1]"""
        return self.gaussians["levels"]

    @property
    def get_extra_levels(self):
        """[N_anchors,]"""
        return self.gaussians["extra_levels"]

    def _add_optimizer_after_backward_hook_if_available(self, optimizer, pl_module):
        hook = getattr(optimizer, "on_after_backward", None)
        if hook is None:
            return
        pl_module.on_after_backward_hooks.append(hook)

    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(opacities)

    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(opacities)

    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales)

    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.log(scales)

    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(rotations)

    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return rotations

    def pre_activate_all_properties(self):
        pass

    @property
    def camera_infos(self) -> torch.Tensor:
        return self._camera_infos

    @property
    def max_level(self) -> int:
        return self._max_level.item()

    @property
    def optimize_from_level(self) -> int:
        return self._optimize_from_level.item()

    @property
    def activate_level(self) -> int:
        if getattr(self, "_activate_level", None) is None:
            return self.optimize_from_level
        return self._activate_level

    @property
    def standard_dist(self) -> float:
        return self._standard_dist.item()

    @property
    def voxel_size(self) -> float:
        return self._voxel_size.item()

    @property
    def grid_origin(self) -> torch.Tensor:
        return self._grid_origin

    @property
    def visibility_threshold(self) -> float:
        return self._visibility_threshold.item()

    @property
    def coarse_intervals(self):
        if getattr(self, "_coarse_intervals", None) is None:
            if self.config.optimization.progressive:
                self._coarse_intervals: List[float] = OctreeUtils.get_coarse_intervals(
                    num_level=self.max_level - self.optimize_from_level + 1,
                    coarse_iter=self.config.optimization.coarse_iter,
                    coarse_factor=self.config.optimization.coarse_factor,
                )
            else:
                self._coarse_intervals = []
        return self._coarse_intervals

    def load_state_dict(self, state_dict, strict=True):
        if "_camera_infos" in state_dict.keys():
            camera_infos = self._camera_infos.new_zeros(state_dict["_camera_infos"].shape)
            self.register_buffer("_camera_infos", camera_infos)
        return super().load_state_dict(state_dict, strict)
