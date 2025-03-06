import math
from dataclasses import dataclass, field
from typing import Dict, Literal

import lightning
import torch
import torch.nn as nn

from internal.cameras.cameras import Camera, Cameras
from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import build_scaling_rotation, inverse_sigmoid, strip_symmetric
from myimpl.scaffold_gaussian.utils import knn, map_to_int_level_factory, xyz_grid_mapping_factory


@dataclass
class OpimizationConfig:
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
                "max_steps": 40_000,
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

    dist2level: Literal["floor", "round", "progressive"] = "round"

    base_layer: int = 11
    """The base layer of the octree"""

    default_voxel_size: float = 0.02

    progressive: bool = True
    """Unknown"""

    extend_ratio: float = 0.1
    """Extend ratio of the octree grid relative to original point cloud"""

    dist_ratio: float = 0.001
    """Filter distances between camera centers and points within (0, dist_ratio) and (1-dist_ratio, 1)"""

    levels: int = -1

    init_level: int = -1

    visibility_threshold: float = 0.01

    coarse_factor: float = 1.5

    coarse_iter: int = 10_000

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
            "levels",  # [N, 1]
            "extra_levels",  # [N_anchors,]
        ]
        self._names = tuple(names)

        self.map_to_int_level = map_to_int_level_factory(self.config.dist2level)

    def get_property_names(self):
        return self._names

    def set_octree_properties(self, points: torch.Tensor, cameras: Cameras, *args, **kwargs):
        cam_centers = cameras.camera_center
        cam_scales = cam_centers.new_ones((cam_centers.shape[0],))

        # calculate levels and register
        levels, init_level, standard_dist = self.get_levels_by_distances(
            points, camera_centers=cam_centers, camera_scales=cam_scales
        )
        self.levels: torch.Tensor; self.init_level: torch.Tensor; self.standard_dist: torch.Tensor  # fmt: skip
        self.register_buffer("levels", levels)
        self.register_buffer("init_level", init_level)
        self.register_buffer("standard_dist", standard_dist)

        # set coarse intervals
        self.set_coarse_intervals()

        # calculate voxel grid and register
        voxel_size, init_pos = self.get_voxel_grid(points)
        self.voxel_size: torch.Tensor
        self.register_buffer("voxel_size", voxel_size)
        self.xyz2grid, self.grid2xyz = xyz_grid_mapping_factory(init_pos=init_pos, padding=self.config.padding)

        # get weed_out function
        self.weed_out = self.weed_out_factory(cam_centers.clone().detach(), cam_scales.clone().detach())

        # get visibility_threshold
        vis_thresh = self.config.visibility_threshold
        if vis_thresh < 0.0:
            positions, levels = self.octree_sample(points)
            vis_thresh, _ = self.weed_out(positions, levels, 0.0)
        self.visibility_threshold: torch.Tensor
        self.register_buffer("visibility_threshold", torch.tensor(vis_thresh))

    def before_setup_set_properties_from_pcd(
        self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs
    ):
        pass

    def setup_from_pcd(self, xyz, rgb, cameras: Cameras, *args, **kwargs):
        points = torch.from_numpy(xyz).to(cameras[0].device).float()
        self.set_octree_properties(points, cameras)

        positions, levels = self.octree_sample(points)
        _, weed_mask = self.weed_out(positions, levels, self.visibility_threshold)
        fused_point_cloud, levels = positions[weed_mask], levels[weed_mask]

        n_anchors, n_offsets = fused_point_cloud.shape[0], self.config.n_offsets

        offsets = fused_point_cloud.new_zeros((n_anchors, n_offsets, 3))
        # dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)
        # scales = self.scale_inverse_activation((torch.sqrt(dist2))[..., None].repeat(1, 6))
        dist = knn(fused_point_cloud, 4)[:, 1:].mean(dim=-1)
        scales = self.scale_inverse_activation(dist).reshape(-1, 1).repeat(1, 6)
        # scales = self.scale_inverse_activation(
        #     dist / (0.5 * self.voxel_size / (float(self.config.fork) ** levels.float()))
        # )
        # scales = scales.reshape(-1, 1).repeat(1, 6)

        anchors = nn.Parameter(fused_point_cloud, requires_grad=True)
        offsets = nn.Parameter(offsets, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)
        levels = nn.Parameter(levels.unsqueeze(1), requires_grad=False)
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
        self.register_buffer("levels", torch.tensor(0, dtype=torch.int))
        self.register_buffer("init_level", torch.tensor(0, dtype=torch.int))
        self.register_buffer("standard_dist", torch.tensor(0, dtype=torch.float))
        self.register_buffer("voxel_size", torch.tensor(0, dtype=torch.float))
        self.register_buffer("visibility_threshold", torch.tensor(0, dtype=torch.float))

        anchors = torch.zeros((n, 3)).float()
        offsets = torch.zeros((n, self.config.n_offsets, 3)).float()
        scales = torch.zeros((n, 6)).float()
        levels = torch.zeros((n, 1)).int()
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

        # anchors and offsets
        # anchors_lr_init = optimization_config.anchors_lr_init * spatial_lr_scale
        # anchors_optimizer = optimizer_factory.instantiate(
        #     [{"params": [self.gaussians["means"]], "name": "means"}],
        #     lr=anchors_lr_init,
        #     eps=1e-15,
        # )
        # self._add_optimizer_after_backward_hook_if_available(anchors_optimizer, module)
        # optimization_config.anchors_lr_scheduler.lr_final *= spatial_lr_scale
        # anchors_scheduler = optimization_config.anchors_lr_scheduler.instantiate().get_scheduler(
        #     anchors_optimizer,
        #     anchors_lr_init,
        # )

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

    def set_coarse_intervals(self):
        self.coarse_intervals = []
        num_level = (self.levels - 1 - self.init_level).item()
        if num_level > 0:
            q = 1.0 / self.config.coarse_factor
            a1 = self.config.coarse_iter * (1 - q) / (1 - q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q**i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def weed_out_factory(self, camera_centers: torch.Tensor, camera_scales: torch.Tensor):
        camera_scales = camera_scales.to(camera_centers)

        def weed_out(positions: torch.Tensor, levels: torch.Tensor, vis_thresh: float, use_chunk=True):
            nonlocal camera_centers, camera_scales

            if len(positions) == 0:
                return (positions.new_zeros((0,), dtype=torch.float), positions.new_zeros((0,), dtype=torch.bool))

            if camera_centers.device != positions.device:
                camera_centers, camera_scales = camera_centers.to(positions), camera_scales.to(positions)

            if use_chunk:
                chunk_size = 4096
                count = positions.new_zeros((positions.shape[0],))
                for st in range(0, len(positions), chunk_size):
                    ed = min(len(positions), st + chunk_size)
                    _anchor, _level = positions[st:ed].reshape(-1, 1, 3), levels[st:ed].reshape(-1, 1)
                    dists = torch.sqrt(torch.sum((_anchor - camera_centers.reshape(-1, 3)) ** 2, dim=-1))
                    pred_levels = torch.log2(self.standard_dist / dists) / math.log2(self.config.fork)
                    int_level, *_ = self.map_to_int_level(pred_levels, self.levels - 1)
                    # if anchor level is lower than level pred by camera
                    # then the anchor is coarse and visible
                    count[st:ed].copy_((_level <= int_level).sum(dim=-1).float())
                count /= len(camera_centers)
            else:
                dists = torch.sqrt(torch.sum((positions.reshape(-1, 1, 3) - camera_centers.reshape(-1, 3)) ** 2))
                pred_levels = torch.log2(self.standard_dist / dists) / math.log2(self.config.fork)
                count = (levels.reshape(-1, 1) <= int_level).sum(dim=-1).float()
                count /= len(camera_centers)
            weed_mask = count > (vis_thresh.to(count) if isinstance(vis_thresh, torch.Tensor) else vis_thresh)
            mean_visibility = torch.mean(count)

            return mean_visibility, weed_mask

        return weed_out

    def get_levels_by_distances(self, points: torch.Tensor, camera_centers: torch.Tensor, camera_scales: torch.Tensor):
        points = points.to(camera_centers)
        all_dist = torch.tensor([]).to(camera_centers)
        for cam_center, cam_scale in zip(camera_centers, camera_scales):
            dist = torch.sqrt(torch.sum((points - cam_center) ** 2, dim=1))
            dist_max = torch.quantile(dist, 1 - self.config.dist_ratio)
            dist_min = torch.quantile(dist, self.config.dist_ratio)
            new_dist = torch.tensor([dist_max, dist_min]).float() * cam_scale
            all_dist = torch.cat([all_dist, new_dist], dim=0)
        dist_max = torch.quantile(all_dist, 1 - self.config.dist_ratio)
        dist_min = torch.quantile(all_dist, self.config.dist_ratio)

        if self.config.levels < 0:
            levels = torch.round(torch.log2(dist_max / dist_min) / math.log2(self.config.fork)).int() + 1
        else:
            levels = torch.tensor(self.config.levels, dtype=torch.int)

        if self.config.init_level < 0:
            init_level = (levels / 2).int()
        else:
            init_level = torch.tensor(self.config.init_level, dtype=torch.int)

        return levels, init_level, dist_max

    def get_voxel_grid(self, points: torch.Tensor):
        box_min, box_max = torch.min(points, dim=0).values, torch.max(points, dim=0).values
        extend_min = box_min - (box_max - box_min) * self.config.extend_ratio
        extend_max = box_max + (box_max - box_min) * self.config.extend_ratio
        box_d = torch.max(extend_max - extend_min)

        if self.config.base_layer < 0:
            self.config.base_layer = (
                torch.round(torch.log2(box_d / self.config.default_voxel_size)).int() - (self.levels // 2) + 1
            ).item()
        voxel_size = box_d / (float(self.config.fork) ** self.config.base_layer)
        return voxel_size, extend_min  # box_min.new_ones((3,)) * box_min

    def octree_sample(self, points: torch.Tensor):
        positions = torch.empty(0, 3).float()
        levels = torch.empty(0).int()
        for cur_level in range(self.levels):
            cur_size = self.voxel_size / (float(self.config.fork) ** cur_level)
            # voxelize, round to nearest grid then padding
            new_positions = self.grid2xyz(torch.unique(self.xyz2grid(points, cur_size), dim=0), cur_size)
            new_level = levels.new_ones((new_positions.shape[0])) * cur_level
            positions = torch.concat((positions, new_positions), dim=0)
            levels = torch.concat((levels, new_level), dim=0)
        return positions, levels

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
