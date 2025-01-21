import math
import os
import os.path as osp
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import pyplot as plt
from sympy import npartitions
from tqdm.auto import tqdm

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera, Cameras
from internal.dataset import CacheDataLoader
from internal.models.vanilla_gaussian import VanillaGaussian, VanillaGaussianModel
from internal.renderers.renderer import Renderer
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.partitioning_utils import (
    MinMaxBoundingBox,
    MinMaxBoundingBoxes,
    PartitionableScene,
    PartitionCoordinates,
    Partitioning,
    SceneBoundingBox,
    SceneConfig,
)


@dataclass
class CityGSSceneConfig(SceneConfig):
    origin: torch.Tensor = None

    partition_size: float = None

    partition_dim: torch.Tensor = torch.tensor([2, 4, 1])  # [3]
    """ block numbers along x-, y-, and z-axis. """

    num_gaussians_per_partition_threshold: int = 25_000
    gaussian_bbox_enlarge_step: torch.Tensor = torch.tensor([0.01, 0.01, 0.01])

    location_based_enlarge: float = 0.2
    visibility_based_distance: float = 0.0
    visibility_based_partition_enlarge: float = 0.0
    visibility_threshold: float = 0.25


@dataclass
class CityGSScene(PartitionableScene):
    scene_config: CityGSSceneConfig
    radius_bounding_box: MinMaxBoundingBox = None
    """ Need to be saved for subsequent contraction calculation """
    enlarged_gaussian_bounding_boxes: MinMaxBoundingBoxes = None

    def get_radius_box_by_cameras_and_points(
        self,
        points: torch.Tensor,
        cameras: Cameras,
        manhattan_trans: Optional[torch.Tensor] = None,
    ):
        """
        Input un-reoriented cameras, and manhattan_trans
        """
        # calculate focus point
        camera_centers: torch.Tensor = cameras.camera_center  # [N, 3]
        focal_axes: torch.Tensor = cameras.R[:, -1, :]  # [N, 3]
        Ms = torch.eye(3) - focal_axes.unsqueeze(-1) * focal_axes.unsqueeze(-2)
        MtMs = Ms.transpose(-1, -2) @ Ms
        A, b = MtMs.mean(dim=0), torch.bmm(MtMs, camera_centers.unsqueeze(-1)).squeeze(-1).mean(dim=0)
        # solve Ax=b
        focus_point = torch.linalg.inv(A) @ b  # [3]
        focus_point_transformed = manhattan_trans[:3, :3] @ focus_point + manhattan_trans[:3, -1]

        camera_centers_transformed = camera_centers @ manhattan_trans[:3, :3].T + manhattan_trans[:3, -1]
        camera_centers_centered = camera_centers_transformed - focus_point_transformed
        radius = torch.median(torch.abs(camera_centers_centered), dim=0).values
        if radius.min() / radius.max() < 0.02:
            # don't contract in the dimension. probably z-axis.
            dim = torch.argmin(radius)
            radius[dim] = 0.5 * (points[:, dim].max() - points[:, dim].min())
        self.radius_bounding_box = MinMaxBoundingBox(
            min=focus_point_transformed - radius,
            max=focus_point_transformed + radius,
        )

        # calculate origin, partition size and scene bounding box
        scene_bbox_min = torch.quantile(points, 0.01, dim=0)
        scene_bbox_max = torch.quantile(points, 0.99, dim=0)

        self.scene_config.origin = focus_point_transformed[:2]
        self.scene_config.partition_size = (
            ((scene_bbox_max - scene_bbox_min) / self.scene_config.partition_dim)[:2].max().item()
        )
        self.scene_bounding_box = SceneBoundingBox(
            MinMaxBoundingBox(min=scene_bbox_min, max=scene_bbox_max),
            n_partitions=self.scene_config.partition_dim,
            origin_partition_offset=None,
        )
        return self.radius_bounding_box

    def bound_coordinates_by_scene_bbox(self, coordinates):
        coordinates = torch.maximum(coordinates, self.scene_bounding_box.bounding_box.min)
        coordinates = torch.minimum(coordinates, self.scene_bounding_box.bounding_box.max)
        return coordinates

    def contract(
        self,
        points: torch.Tensor,
        ord: float = 2,
        eps: float = 1e-6,
        inversed: bool = False,
    ):
        if inversed:
            norm = torch.linalg.norm(points, ord=ord, dim=-1)
            equal_to_2 = torch.abs(points) + eps > 2.0  # torch.abs(torch.abs(points) - 2.0) < eps
            between_1_and_2 = torch.logical_and(norm > 1.0, ~equal_to_2.any(dim=-1))
            is_positive = points[equal_to_2] > 0.0

            scale = norm.new_ones(norm.shape)
            scale[between_1_and_2] = 1.0 / (norm[between_1_and_2] * (2 - norm[between_1_and_2]))
            points_uncontracted = points * scale.unsqueeze(-1)
            points_unnormalized = (
                (points_uncontracted + 1.0) / 2.0 * (self.radius_bounding_box.max - self.radius_bounding_box.min)
            ) + self.radius_bounding_box.min
            points_unnormalized[equal_to_2] = torch.where(is_positive, torch.inf, -torch.inf)
            return points_unnormalized
        else:
            points_normalized = (
                (points - self.radius_bounding_box.min) / (self.radius_bounding_box.max - self.radius_bounding_box.min)
            ) * 2 - 1  # normalize points in radius to [-1, 1]
            norm: torch.Tensor = torch.linalg.norm(points_normalized, ord=ord, dim=-1)
            mask = norm > 1.0
            scale = norm.new_ones(norm.shape)
            scale[mask] = (2.0 - 1.0 / norm[mask]) / norm[mask]
            return points_normalized * scale.unsqueeze(-1)

    def build_partition_coordinates(self):
        """
        Build partition coordinates in contracted space. Get mask in contracted space when merging.
        """
        grid_x, grid_y, grid_z = torch.meshgrid(
            *[torch.arange(k) for k in self.scene_config.partition_dim], indexing="ij"
        )
        id_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).long()
        xyz_min = id_tensor.float() / self.scene_config.partition_dim
        xyz_min = xyz_min * 4 - 2  # [0, 1] -> [-2, 2]
        xyz_max = (id_tensor.float() + 1) / self.scene_config.partition_dim
        xyz_max = xyz_max * 4 - 2

        xyz_min_uncontracted = self.contract(xyz_min, ord=torch.inf, inversed=True)
        xyz_max_uncontracted = self.contract(xyz_max, ord=torch.inf, inversed=True)
        xyz_min_regularized = self.bound_coordinates_by_scene_bbox(xyz_min_uncontracted)
        xyz_max_regularized = self.bound_coordinates_by_scene_bbox(xyz_max_uncontracted)
        self.partition_coordinates = PartitionCoordinates(
            id=id_tensor, xy=xyz_min_regularized, size=xyz_max_regularized - xyz_min_regularized
        )
        return self.partition_coordinates

    def get_enlarged_gaussian_bboxes(self, points: torch.Tensor):
        """
        input points are transformed. enlarge in contracted coordinates.
        """
        points_contracted = self.contract(points, ord=torch.inf)

        bboxes = self.partition_coordinates.get_bounding_boxes()
        enlarged_min, enlarged_max = [], []
        self.gaussians_in_partitions = torch.zeros((len(self.partition_coordinates), len(points)), dtype=torch.bool)

        for partition_id in range(len(self.partition_coordinates)):
            bbox: MinMaxBoundingBox = bboxes[partition_id]
            bbox_contracted = MinMaxBoundingBox(
                min=self.contract(bbox.min, ord=torch.inf),
                max=self.contract(bbox.max, ord=torch.inf),
            )
            gaussians_in_block = (
                torch.prod(points_contracted > bbox_contracted.min, dim=-1)
                * torch.prod(points_contracted < bbox_contracted.max, dim=-1)
            ) > 0
            num_gaussians = gaussians_in_block.sum()
            while num_gaussians < self.scene_config.num_gaussians_per_partition_threshold:
                bbox_contracted.min -= self.scene_config.gaussian_bbox_enlarge_step
                bbox_contracted.max += self.scene_config.gaussian_bbox_enlarge_step
                gaussians_in_block = (
                    torch.prod(points_contracted > bbox_contracted.min, dim=-1)
                    * torch.prod(points_contracted < bbox_contracted.max, dim=-1)
                ) > 0
                num_gaussians = gaussians_in_block.sum()
            self.gaussians_in_partitions[partition_id].copy_(gaussians_in_block)

            # fix bug: the contract mapping is not order-perserving
            enlarged_min.append(self.contract(bbox_contracted.min, ord=torch.inf, inversed=True))
            enlarged_max.append(self.contract(bbox_contracted.max, ord=torch.inf, inversed=True))

        self.enlarged_gaussian_bounding_boxes = MinMaxBoundingBoxes(
            min=self.bound_coordinates_by_scene_bbox(torch.stack(enlarged_min, 0)),
            max=self.bound_coordinates_by_scene_bbox(torch.stack(enlarged_max, 0)),
        )
        return self.enlarged_gaussian_bounding_boxes

    def camera_center_based_partition_assignment(self):
        bounding_boxes = self.partition_coordinates.get_bounding_boxes(
            enlarge=self.scene_config.location_based_enlarge,
        )
        self.is_camera_in_partition = Partitioning.is_in_bounding_boxes(
            bounding_boxes=MinMaxBoundingBoxes(min=bounding_boxes.min[..., :2], max=bounding_boxes.max[..., :2]),
            coordinates=self.camera_centers[..., :2],
        )
        return self.is_camera_in_partition

    @staticmethod
    def is_in_partition(coordinates: torch.Tensor, bbox: MinMaxBoundingBox):
        xy_min, xy_max = bbox.min, bbox.max
        is_gt_min = torch.prod(torch.ge(coordinates, xy_min), dim=-1)  # [N_coordinates]
        is_le_max = torch.prod(torch.le(coordinates, xy_max), dim=-1)  # [N_coordinates]
        is_in_partition = is_gt_min * is_le_max != 0
        return is_in_partition

    def calculate_camera_visibilities(
        self,
        coarse_model: VanillaGaussianModel,
        renderer: VanillaRenderer,
        cameras: Cameras,
        device: Any = "cpu",
        bg_color: Tuple[float] = (0.0, 0.0, 0.0),
    ):
        """
        For each partition:
            1. build a incomplete gaussian model
            2. render two image
            3. calculate ssim between completely rendered image and incomplete one
        """
        from internal.utils.ssim import ssim

        self.camera_visibilities = torch.zeros(
            (len(self.partition_coordinates), len(cameras)),
            dtype=torch.float32,
        )
        torch.cuda.empty_cache()
        # build incomplete gaussian model
        complete_properties = coarse_model.properties
        for partition_idx in range(len(self.gaussians_in_partitions)):
            mask = ~self.gaussians_in_partitions[partition_idx].to(device)
            incomplete_properties = {k: v[mask] for k, v in complete_properties.items()}

            with torch.no_grad():
                for camera_idx, camera in enumerate(tqdm(cameras)):
                    camera = camera.to_device(device=device)
                    # completely rendered
                    coarse_model.properties = complete_properties
                    completely_rendered = renderer(
                        camera,
                        coarse_model,
                        torch.tensor(bg_color).to(device),
                    )["render"]

                    # incompletely rendered
                    coarse_model.properties = incomplete_properties
                    incompletely_rendered = renderer(
                        camera,
                        coarse_model,
                        torch.tensor(bg_color).to(device),
                    )["render"]

                    self.camera_visibilities[partition_idx, camera.idx].copy_(
                        1 - ssim(incompletely_rendered.unsqueeze(0), completely_rendered.unsqueeze(0))
                    )
        coarse_model.properties = complete_properties
        return self.camera_visibilities

    def visibility_based_partition_assignment(self):
        self.is_partitions_visible_to_cameras = self.camera_visibilities > self.scene_config.visibility_threshold
        return self.is_partitions_visible_to_cameras

    def save_plot(self, func: Callable, path: str, *args, **kwargs):
        fig, ax = plt.subplots()
        func(ax, *args, **kwargs)
        plt.savefig(path, dpi=600)
        plt.close(fig)
