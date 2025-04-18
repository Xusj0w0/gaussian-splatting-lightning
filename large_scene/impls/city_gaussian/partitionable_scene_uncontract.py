import json
import math
import os
import os.path as osp
import pickle
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from jsonargparse import ArgumentParser
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, box
from tqdm import tqdm

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.dataparser import ImageSet
from internal.models.vanilla_gaussian import (VanillaGaussian,
                                              VanillaGaussianModel)
from internal.renderers.renderer import Renderer
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.graphics_utils import BasicPointCloud
from internal.utils.partitioning_utils import (MinMaxBoundingBox,
                                               MinMaxBoundingBoxes,
                                               PartitionCoordinates,
                                               Partitioning, SceneBoundingBox)
from internal.utils.sh_utils import eval_sh

from ..base.partitionable_scene import (PartitionableScene,
                                        PartitionableSceneConfig)
from .partitionable_scene import CityScene

__all__ = ["UncontractCitySceneConfig", "UncontractCityScene"]


@dataclass
class UncontractCitySceneConfig(PartitionableSceneConfig):
    down_sample_factor: int = 4
    """ down sample factor when coarse training """

    train_config: str = None
    """ path to config file for coarse training """

    visibility_threshold: float = field(default=0.05)

    def instantiate(self):
        return UncontractCityScene(self)


@dataclass
class UncontractCityScene(CityScene):
    scene_config: UncontractCitySceneConfig = field(default_factory=lambda: UncontractCitySceneConfig())

    gaussian_bbox_enlarge_step: torch.Tensor = field(init=False)

    radius_bounding_box: MinMaxBoundingBox = field(init=False)

    gaussians_in_partitions: torch.Tensor = field(default=None, init=False)
    """ indicate each gaussian in which partition, [N_partition, N_gaussian] """

    def __post_init__(self):
        PartitionableScene.__post_init__(self)

    def partition(self, dataset_path: str, output_path: str):
        device = torch.device("cuda")
        # coarse training
        coarse_model, renderer, image_set, ckpt = self.load_coarse_model(dataset_path, output_path, device)

        # calculate points' xyz
        camera_centers = image_set.cameras.camera_center
        camera_centers_transformed = camera_centers @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        self.camera_centers = camera_centers_transformed[..., :2]

        means = coarse_model.get_xyz.detach().clone().cpu()
        means_transformed = means @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]

        # get bounding boxes
        self.get_bounding_box_by_points(means_transformed)
        self.get_bounding_box_by_camera_centers()
        # get scene bbox by enlarged camera centers
        self.get_scene_bounding_box()
        # extend to scene bbox
        self.build_partition_coordinates()
        # get bbox bounded by camera centers (for location based assignment and plot)
        bounded_coordinates = self.bound_partition_coordinates_by_camera_bbox()
        partition_coordinates = self.partition_coordinates

        # assign cameras to partitions
        # location based assignment, use bounded coordinates
        self.partition_coordinates = bounded_coordinates
        PartitionableScene.camera_center_based_partition_assignment(self)
        # visibility calculation, use orig coordinates
        self.partition_coordinates = partition_coordinates
        # render image with one of the partitions removed
        bg_color = ckpt["hyper_parameters"]["background_color"]
        self.gaussians_in_partitions = Partitioning.is_in_bounding_boxes(
            self.partition_coordinates.get_bounding_boxes(), means_transformed[..., :2]
        )
        self.calculate_camera_visibilities(coarse_model, renderer, image_set.cameras, device=device, bg_color=bg_color)
        # assign cameras based on visibilities
        self.visibility_based_partition_assignment()
        self.visibility_based_gaussian_assignment(coarse_model, image_set.cameras, device=device)

        self.partition_coordinates = bounded_coordinates
        # load point cloud for visualization
        point_cloud = self.load_point_cloud(dataset_path)
        points3D_transformed = (
            point_cloud.points.to(self.manhattan_trans) @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        )
        self.save_plots(
            output_path, BasicPointCloud(points=points3D_transformed, colors=point_cloud.colors, normals=None)
        )
        self.partition_coordinates = partition_coordinates
        self.save_partitioning_results(output_path, image_set, coarse_model)

    @classmethod
    def is_in_partition(
        cls,
        coordinates: torch.Tensor,
        partition_bbox: MinMaxBoundingBox,
        manhattan_trans: torch.Tensor,
        *args,
        **kwargs,
    ):
        return PartitionableScene.is_in_partition(coordinates, partition_bbox, manhattan_trans)

    def get_scene_bounding_box(self):
        """
        Orgininal implementation use grids to cover the point-based bbox.
        We divide enlarged camera bounding box as scene bounding box A;
        set origin to be center of A;
        get partition_size with A and partition_dim;
        refine the bbox as the final scene bbox.
        """
        # bug: xyz<0 will add
        # scene_bbox = Partitioning.get_bounding_box_by_camera_centers(
        #     self.camera_centers, enlarge=self.scene_config.scene_bbox_enlarge_by_camera_bbox
        # )
        size = self.camera_center_based_bounding_box.max - self.camera_center_based_bounding_box.min
        scene_bbox = MinMaxBoundingBox(
            min=self.camera_center_based_bounding_box.min - self.scene_config.scene_bbox_enlarge_by_camera_bbox * size,
            max=self.camera_center_based_bounding_box.max + self.scene_config.scene_bbox_enlarge_by_camera_bbox * size,
        )
        self.scene_config.origin = 0.5 * (scene_bbox.min + scene_bbox.max)

        size = scene_bbox.max - scene_bbox.min
        size_per_partition = size / self.partition_dim[:2]
        self.scene_config.partition_size = size_per_partition.max().item()

        size = self.scene_config.partition_size * self.partition_dim[:2]
        scene_bbox = MinMaxBoundingBox(
            min=self.scene_config.origin - 0.5 * size, max=self.scene_config.origin + 0.5 * size
        )
        self.scene_bounding_box = SceneBoundingBox(
            bounding_box=scene_bbox,
            n_partitions=self.partition_dim[:2],
            origin_partition_offset=None,
        )
        return self.scene_bounding_box

    def bound_partition_coordinates_by_camera_bbox(self):
        bbox = self.camera_center_based_bounding_box
        ids, xys, sizes = [], [], []
        for id, xy, size in self.partition_coordinates:
            ids.append(id)
            _bbox = MinMaxBoundingBox(min=xy, max=xy + size)
            bounded_box = MinMaxBoundingBox(
                min=torch.maximum(_bbox.min, bbox.min),
                max=torch.minimum(_bbox.max, bbox.max),
            )
            xys.append(bounded_box.min)
            sizes.append(bounded_box.max - bounded_box.min)
        return PartitionCoordinates(
            id=torch.stack(ids, dim=0),
            xy=torch.stack(xys, dim=0),
            size=torch.stack(sizes, dim=0),
        )

    def balanced_camera_based_division(self):
        """
        Reference VastGaussian implementation: https://github.com/kangpeilun/VastGaussian
        Correspond to Camera_position_based_region_division() in VastGS
        1. Divide cameras along x-axis;
        2. Divide cameras along y-axis;
        """
        assert self.camera_centers is not None, "Camera centers are not available."
        num_cameras = len(self.camera_centers)
        x_dim, y_dim, z_dim = self.partition_dim.long().tolist()

        # 1. Divide cameras along x-axis
        # diff: VastGaussian uses floor, and merge remaining cameras into the last partition
        num_cameras_per_column = math.ceil(num_cameras / x_dim)
        camera_positions = deepcopy(self.camera_centers)
        _, x_sort_indices = torch.sort(camera_positions[:, 0], dim=0)
        partition_dict: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for i, x_st in enumerate(range(0, num_cameras, num_cameras_per_column)):
            x_ed = min(x_st + num_cameras_per_column, num_cameras)
            x_mid_camera_id = x_sort_indices[-1] if x_ed < num_cameras else None
            camera_indices_in_column = x_sort_indices[x_st:x_ed]
            camera_centers_in_column = camera_positions[camera_indices_in_column]

            # 2. Divide cameras along y-axis
            _, y_sort_indices = torch.sort(camera_centers_in_column[:, 1], dim=0)
            num_cameras_in_column = len(camera_centers_in_column)
            num_cameras_per_partition = math.ceil(num_cameras_in_column / y_dim)
            for j, y_st in enumerate(range(0, num_cameras_in_column, num_cameras_per_partition)):
                y_ed = min(y_st + num_cameras_per_partition, num_cameras_in_column)
                camera_indices_in_partition = camera_indices_in_column[y_sort_indices[y_st:y_ed]]
                y_mid_camera_id = camera_indices_in_partition[-1] if y_ed < num_cameras_in_column else None

                partition_dict[f"{i}_{j}"] = {
                    "camera_indices": camera_indices_in_partition,
                    "x_mid_camera_id": x_mid_camera_id,
                    "y_mid_camera_id": y_mid_camera_id,
                }

        return partition_dict

    def refine_region_division(self, partition_dict: Dict[str, Dict[str, Optional[torch.Tensor]]]):
        """
        Reference VastGaussian implementation: https://github.com/kangpeilun/VastGaussian
        Correspond to refine_ori_bbox() in VastGS
        Original implementation use camera position as boundary.
        We use the average of min-max range as boundary.
        """
        x_dim, y_dim, z_dim = self.partition_dim.long().tolist()
        camera_positions = deepcopy(self.camera_centers)

        # Calculate partition bbox by cameras in partition
        # Stitching result contains gaps.
        bbox_dict: Dict[str, MinMaxBoundingBox] = {}
        for partition_idx, camera_id_dict in partition_dict.items():
            camera_indices = camera_id_dict["camera_indices"]
            camera_positions_in_partition = camera_positions[camera_indices]
            bbox_dict[partition_idx] = MinMaxBoundingBox(
                min=torch.min(camera_positions_in_partition[:, :2], dim=0).values,
                max=torch.max(camera_positions_in_partition[:, :2], dim=0).values,
            )

        # Refine along y-axis
        for i in range(x_dim):
            for j in range(y_dim - 1):
                bottom_partition_id = f"{i}_{j}"
                up_partition_id = f"{i}_{j+1}"
                # mid_camera_id = partition_dict[bottom_partition_id]["y_mid_camera_id"]
                # y_mid = camera_positions[mid_camera_id, 1]
                y_mid = 0.5 * (bbox_dict[bottom_partition_id].max[1] + bbox_dict[up_partition_id].min[1])
                bbox_dict[bottom_partition_id].max[1] = y_mid
                bbox_dict[up_partition_id].min[1] = y_mid

                if j == 0:
                    bbox_dict[bottom_partition_id].min[1] = self.scene_bounding_box.bounding_box.min[1]
                if j == y_dim - 2:
                    bbox_dict[up_partition_id].max[1] = self.scene_bounding_box.bounding_box.max[1]

        # Refine along x-axis
        for j in range(y_dim):
            for i in range(x_dim - 1):
                left_partition_id = f"{i}_{j}"
                right_partition_id = f"{i+1}_{j}"
                # mid_camera_id = partition_dict[left_partition_id]["x_mid_camera_id"]
                # x_mid = camera_positions[mid_camera_id, 0]
                x_mid = 0.5 * (bbox_dict[left_partition_id].max[0] + bbox_dict[right_partition_id].min[0])
                bbox_dict[left_partition_id].max[0] = x_mid
                bbox_dict[right_partition_id].min[0] = x_mid

                if i == 0:
                    bbox_dict[left_partition_id].min[0] = self.scene_bounding_box.bounding_box.min[0]
                if i == x_dim - 2:
                    bbox_dict[right_partition_id].max[0] = self.scene_bounding_box.bounding_box.max[0]

        return bbox_dict

    def get_extra_data(self):
        extra_data = super().get_extra_data()
        extra_data.update(
            {
                "up": torch.linalg.inv(self.manhattan_trans)[:3, 1],
                "rotation_transform": self.manhattan_trans,
                "gaussians_in_partitions": self.gaussians_in_partitions,
            }
        )
        return extra_data

    def build_partition_coordinates(self):
        partition_dict = self.balanced_camera_based_division()
        bbox_dict = self.refine_region_division(partition_dict)
        id_tensor, xy_tensor, sz_tensor = (
            torch.empty([0, 2], dtype=torch.int),
            torch.empty([0, 2], dtype=torch.float32),
            torch.empty([0, 2], dtype=torch.float32),
        )
        for partition_id, bbox in bbox_dict.items():
            _id = torch.tensor([[int(s) for s in partition_id.split("_")]])
            _xy = deepcopy(bbox.min).unsqueeze(0)
            _sz = deepcopy(bbox.max - bbox.min).unsqueeze(0)
            id_tensor = torch.cat([id_tensor, _id], 0)
            xy_tensor = torch.cat([xy_tensor, _xy], 0)
            sz_tensor = torch.cat([sz_tensor, _sz], 0)

        self.partition_coordinates = PartitionCoordinates(id=id_tensor, xy=xy_tensor, size=sz_tensor)

        return self.partition_coordinates
