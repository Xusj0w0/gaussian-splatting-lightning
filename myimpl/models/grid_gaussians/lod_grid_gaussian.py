from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import lightning
import numpy as np
import torch
import torch.nn as nn

from internal.cameras.cameras import Camera, Cameras
from internal.utils.general_utils import inverse_sigmoid
from myimpl.models.grid_gaussians.grid_gaussian import GridGaussian, GridGaussianModel, GridOptimizationConfig
from myimpl.models.grid_gaussians.utils import GridGaussianUtils


@dataclass
class LoDGridOptimizationConfig(GridOptimizationConfig):
    progressive: bool = True
    """whether optimize anchor from coarse to fine progressively"""

    coarse_factor: float = 1.5

    coarse_iter: int = 10_000


@dataclass
class LoDGridGaussian(GridGaussian):
    fork: int = 2

    dist2level: Literal["floor", "round", "progressive"] = "floor"

    extend_ratio: float = 0.1
    """Extend ratio of the octree grid relative to original point cloud"""

    dist_ratio: float = 0.001
    """Filter distances between camera centers and points within (0, dist_ratio) and (1-dist_ratio, 1)"""

    default_voxel_size: float = 0.02

    max_level: int = -1

    start_level: int = -1

    visibility_threshold: float = 0.01

    optimization: LoDGridOptimizationConfig = field(default_factory=lambda: LoDGridOptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return LoDGridGaussianModel(self)


class LoDGridGaussianModel(GridGaussianModel):
    """
    buffers: levels, init_level, standard_dist, voxel_size, visibility_threshold
    """

    def __init__(self, config: LoDGridGaussian):
        super().__init__(config)
        self.config = config

        names = [
            "levels",  # [N,]
            "extra_levels",  # [N,]
        ]
        self._names = tuple(list(self._names) + names)

        buffer_names = [
            "_max_level",
            "_start_level",
            "_standard_dist",
            "_visibility_threshold",
        ]
        self._buffer_names = tuple(list(self._buffer_names) + buffer_names)

    def setup_multi_level_grid(self, points: torch.Tensor, camera_infos: torch.Tensor, *args, **kwargs):
        # calculate levels and register
        standard_dist, max_level = GridGaussianUtils.get_levels_by_distances(
            points, camera_infos, self.config.dist_ratio, self.config.fork
        )
        max_level = torch.tensor(self.config.max_level) if self.config.max_level > 0 else max_level
        start_level = torch.tensor(self.config.start_level) if self.config.start_level > 0 else (max_level / 2).int()
        self.register_buffer("_max_level", max_level)
        self.register_buffer("_start_level", start_level)
        self.register_buffer("_standard_dist", standard_dist)

        # calculate voxel grid and register
        voxel_size, grid_origin = GridGaussianUtils.build_multi_level_grid(
            points,
            extend_ratio=self.config.extend_ratio,
            base_layer=self.config.base_layer,
            fork=self.config.fork,
            default_voxel_size=self.config.default_voxel_size,
            max_level=self.max_level,
        )
        self.register_buffer("_voxel_size", voxel_size)
        self.register_buffer("_grid_origin", grid_origin)

        # get visibility_threshold
        vis_thresh = torch.tensor(self.config.visibility_threshold)
        if vis_thresh < 0.0:
            positions, levels = GridGaussianUtils.multi_level_voxelize(
                points,
                voxel_size=self.voxel_size,
                max_level=self.max_level,
                xyz2grid=self.xyz2grid,
                grid2xyz=self.grid2xyz,
                fork=self.config.fork,
            )
            mask = GridGaussianUtils.weed_out_mask_by_level(
                positions,
                levels,
                0.0,
                cam_infos=camera_infos,
                predict_level_fn=self.predict_level,
                int_level_fn=lambda x: self.map_to_int_level(x, self.max_level)[0],
            )
            vis_thresh = torch.mean(mask.float())
        self.register_buffer("_visibility_threshold", vis_thresh)

    def before_setup_properties_from_pcd(
        self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs
    ):
        pass

    def setup_from_pcd(self, xyz, rgb, cameras: Cameras, *args, **kwargs):
        points = torch.from_numpy(xyz).to(cameras[0].device).float()
        cam_centers = cameras.camera_center
        camera_infos = torch.cat([cam_centers, cam_centers.new_ones((cam_centers.shape[0], 1))], dim=-1)
        self.setup_multi_level_grid(points, camera_infos)

        positions, levels = GridGaussianUtils.multi_level_voxelize(
            points,
            voxel_size=self.voxel_size,
            max_level=self.max_level,
            xyz2grid=self.xyz2grid,
            grid2xyz=self.grid2xyz,
            fork=self.config.fork,
        )
        weed_mask = GridGaussianUtils.weed_out_mask_by_level(
            positions,
            levels,
            self.visibility_threshold,
            cam_infos=camera_infos,
            predict_level_fn=self.predict_level,
            int_level_fn=lambda x: self.map_to_int_level(x, self.max_level)[0],
        )
        fused_point_cloud, levels = positions[weed_mask], levels[weed_mask]

        property_dict = self.get_init_properties(
            fused_point_cloud=fused_point_cloud,
            levels=levels,
            extra_levels=levels.new_zeros((levels.shape[0],), dtype=torch.float),
            mode="pcd",
        )
        self.before_setup_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        pass

    def setup_from_number(self, n, *args, **kwargs):
        self.register_buffer("_voxel_size", torch.tensor(0, dtype=torch.float))
        self.register_buffer("_grid_origin", torch.zeros((3,), dtype=torch.float))
        self.register_buffer("_max_level", torch.tensor(0, dtype=torch.int))
        self.register_buffer("_start_level", torch.tensor(0, dtype=torch.int))
        self.register_buffer("_standard_dist", torch.tensor(0, dtype=torch.float))
        self.register_buffer("_visibility_threshold", torch.tensor(0, dtype=torch.float))

        property_dict = self.get_init_properties(n=n, mode="number")
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        for name, value in property_dict.items():
            self.set_property(name, value)

    def setup_from_tensors(self, tensors, *args, **kwargs):
        pass

    def training_setup_lod_properties(self, module, *args, **kwargs):
        self._activate_level = self.max_level
        if self.config.optimization.progressive:
            self._activate_level = np.searchsorted(self.coarse_intervals, module.trainer.global_step) + 1 + self.start_level
        module.on_train_batch_end_hooks.append(self.activate_level_update)
        return [], []

    def get_lod_properties(
        self,
        fused_point_cloud: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        tensors: Optional[Tuple[torch.Tensor]] = None,
        mode: Literal["pcd", "number", "tensors"] = "pcd",
        levels: Optional[torch.Tensor] = None,
        extra_levels: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if mode == "pcd":
            assert fused_point_cloud is not None
        elif mode == "number":
            assert n is not None
            levels = torch.zeros((n,), dtype=torch.int)
            extra_levels = torch.zeros((n,), dtype=torch.float)
        elif mode == "tensors":
            pass
        else:
            raise ValueError(f"Unsupported mode {mode}")

        levels = nn.Parameter(levels, requires_grad=False)
        extra_levels = nn.Parameter(extra_levels, requires_grad=False)
        property_dict = {
            "levels": levels,
            "extra_levels": extra_levels,
        }
        return property_dict

    def predict_level(self, dists: torch.Tensor):
        return GridGaussianUtils.predict_level(dists, standard_dist=self.standard_dist, fork=self.config.fork)

    def map_to_int_level(self, pred_level: torch.Tensor, cur_level: int):
        return GridGaussianUtils.map_to_int_level(pred_level, cur_level, dist2level=self.config.dist2level)

    @classmethod
    def activate_level_update(
        cls, outputs, batch, gaussian_model: "LoDGridGaussianModel", global_step, pl_module: lightning.LightningModule
    ):
        if gaussian_model.config.optimization.progressive:
            gaussian_model._activate_level = (
                np.searchsorted(gaussian_model.coarse_intervals, global_step) + 1 + gaussian_model.start_level
            )

    @property
    def get_levels(self) -> torch.Tensor:
        """[N_anchors, 1]"""
        return self.gaussians["levels"]

    @property
    def get_extra_levels(self) -> torch.Tensor:
        """[N_anchors,]"""
        return self.gaussians["extra_levels"]

    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(opacities)

    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(opacities)

    def pre_activate_all_properties(self):
        pass

    @property
    def max_level(self) -> int:
        self._max_level: torch.Tensor
        return self._max_level.item()

    @property
    def start_level(self) -> int:
        self._start_level: torch.Tensor
        return self._start_level.item()

    @property
    def activate_level(self) -> int:
        if getattr(self, "_activate_level", None) is None:
            self._activate_level = self.max_level
        return self._activate_level

    @activate_level.setter
    def activate_level(self, value: int):
        self._activate_level = value

    @property
    def standard_dist(self) -> float:
        return self._standard_dist.item()

    @property
    def visibility_threshold(self) -> float:
        self._visibility_threshold: torch.Tensor
        return self._visibility_threshold.item()

    @property
    def coarse_intervals(self):
        if getattr(self, "_coarse_intervals", None) is None:
            if self.config.optimization.progressive:
                self._coarse_intervals: List[float] = GridGaussianUtils.get_coarse_intervals(
                    num_level=self.max_level - self.start_level + 1,
                    coarse_iter=self.config.optimization.coarse_iter,
                    coarse_factor=self.config.optimization.coarse_factor,
                )
            else:
                self._coarse_intervals = []
        return self._coarse_intervals
