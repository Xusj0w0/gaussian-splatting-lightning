import json
import math
import os
import os.path as osp
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
import torch
from jsonargparse import ArgumentParser
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box
from tqdm import tqdm

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.dataparser import ImageSet
from internal.utils.graphics_utils import BasicPointCloud
from internal.utils.partitioning_utils import (MinMaxBoundingBox,
                                               MinMaxBoundingBoxes,
                                               PartitionCoordinates,
                                               Partitioning, SceneBoundingBox)

from ..base.partitionable_scene import (PartitionableScene,
                                        PartitionableSceneConfig)

__all__ = ["VastScene", "VastSceneConfig"]


@dataclass
class VastSceneConfig(PartitionableSceneConfig):
    min_track_length: int = 3
    """ minimum track length for prefiltering point cloud """

    max_error: float = 2.0
    """ maximum error for prefiltering point cloud """

    def instantiate(self):
        return VastScene(self)


@dataclass
class VastScene(PartitionableScene):

    scene_config: VastSceneConfig = field(default_factory=lambda: VastSceneConfig())

    def partition(self, dataset_path: str, output_path: str):
        image_set, point_cloud = self.load_sparse_model(dataset_path)

        reoriented_camera_centers: torch.Tensor = (
            image_set.cameras.camera_center @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        )
        reoriented_points3D: torch.Tensor = (
            point_cloud.points @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        )
        self.camera_centers = reoriented_camera_centers[..., :2]

        # get bounding boxes
        self.get_bounding_box_by_points(reoriented_points3D)
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
        self.camera_center_based_partition_assignment()
        # cube based visibility calculation, use orig coordinates
        self.partition_coordinates = partition_coordinates
        vertices = self.get_partition_cube_vertices(reoriented_points3D)
        bbox_centers = vertices.reshape(-1, 8, 3).mean(dim=1)
        inversed_manhattan_trans = torch.linalg.inv(self.manhattan_trans)
        vertices_inverse_manhattan_transformed = (
            vertices @ inversed_manhattan_trans[:3, :3].T + inversed_manhattan_trans[:3, -1]
        )
        self.calculate_camera_visibilities(
            point_getter=self.get_point_getter_fn(
                image_set.cameras,
                vertices_inverse_manhattan_transformed,
                bbox_centers,
            ),
            device=image_set.cameras.R.device,
        )
        self.visibility_based_partition_assignment()

        # save plots, use bounded coordinates
        self.partition_coordinates = bounded_coordinates
        self.save_plots(
            output_path,
            BasicPointCloud(points=reoriented_points3D, colors=point_cloud.colors, normals=point_cloud.normals),
        )
        # save results, use orig coordinates
        self.partition_coordinates = partition_coordinates
        self.save_partitioning_results(output_path, image_set)

    def get_extra_data(self):
        extra_data = super().get_extra_data()
        extra_data.update(
            {"up": torch.linalg.inv(self.manhattan_trans)[:3, 1], "rotation_transform": self.manhattan_trans}
        )
        return extra_data

    @staticmethod
    def load_points_from_bin(points3D_file_path: str) -> List[torch.Tensor]:
        with open(points3D_file_path, "rb") as fid:
            num_points = colmap_utils.read_next_bytes(fid, 8, "Q")[0]

            xyzs = []
            rgbs = []
            errors = []
            track_lengths = []

            for p_id in range(num_points):
                binary_point_line_properties = colmap_utils.read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd"
                )
                point3D_id = binary_point_line_properties[0]
                xyz = torch.tensor(binary_point_line_properties[1:4])
                rgb = torch.tensor(binary_point_line_properties[4:7])
                error = torch.tensor(binary_point_line_properties[7])
                track_length = colmap_utils.read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
                _ = colmap_utils.read_next_bytes(
                    fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length
                )
                xyzs.append(xyz)
                rgbs.append(rgb)
                errors.append(error)
                track_lengths.append(torch.tensor(track_length))

        return list(map(lambda x: torch.stack(x, dim=0), [xyzs, rgbs, errors, track_lengths]))

    def load_sparse_model(self, dataset_path: str):
        dataparser_config = Colmap(split_mode="reconstruction", points_from="random")
        dataparser: ColmapDataParser = dataparser_config.instantiate(
            path=dataset_path, output_path=os.getcwd(), global_rank=0
        )
        points3D_path = osp.join(dataparser.detect_sparse_model_dir(), "points3D.bin")
        dataparser_outputs = dataparser.get_outputs()

        image_set: ImageSet = dataparser_outputs.train_set

        xyzs, rgbs, errors, track_lengths = self.load_points_from_bin(points3D_path)
        mask = torch.logical_and(
            torch.ge(track_lengths, self.scene_config.min_track_length),
            torch.le(errors, self.scene_config.max_error),
        )
        return image_set, BasicPointCloud(points=xyzs[mask], colors=rgbs[mask], normals=None)

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

    def visibility_based_partition_assignment(self):
        # [N_partitions, N_cameras]
        self.is_partitions_visible_to_cameras = self.camera_visibilities > self.scene_config.visibility_threshold
        return self.is_partitions_visible_to_cameras

    def get_partition_cube_vertices(self, xyzs: torch.Tensor):
        """
        VastGS implementation extract points in partition coordinates and calculate 3D bounding boxes.
        We take 0.5% points in up and bottom as outliers and remove them.
        """

        def bbox3D_to_eight_points(bbox: MinMaxBoundingBox) -> torch.Tensor:
            assert len(bbox.min) == 3, "Bounding box should be 3-dimensional."
            vertices = bbox.min.new_zeros((8, 3))
            cnt = 0
            for x in [bbox.min[0], bbox.max[0]]:
                for y in [bbox.min[1], bbox.max[1]]:
                    for z in [bbox.min[2], bbox.max[2]]:
                        vertices[cnt] = torch.as_tensor([x, y, z])
                        cnt += 1
            return vertices

        bboxes = self.partition_coordinates.get_bounding_boxes()
        points_in_partitions = Partitioning.is_in_bounding_boxes(bboxes, xyzs[..., :2])
        vertices = xyzs.new_empty((0, 3))
        for i in range(len(self.partition_coordinates)):
            xyzs_in_partition = xyzs[points_in_partitions[i]]
            z_bound = self.get_3D_bounding_box_by_points(points=xyzs_in_partition[..., 2:], outlier_threshold=0.001)
            xy_bound = bboxes[i]
            xyz_bound = MinMaxBoundingBox(
                min=torch.cat([xy_bound.min, z_bound.min]), max=torch.cat([xy_bound.max, z_bound.max])
            )
            vertices = torch.cat([vertices, bbox3D_to_eight_points(xyz_bound)], 0)

        return vertices

    @staticmethod
    def get_3D_bounding_box_by_points(points, enlarge=0, outlier_threshold=0.001):
        xyz_min = torch.quantile(points, outlier_threshold, dim=0)
        xyz_max = torch.quantile(points, 1.0 - outlier_threshold, dim=0)

        if enlarge > 0.0:
            size = xyz_max - xyz_min
            enlarge_size = size * enlarge
            xyz_min -= enlarge_size
            xyz_max += enlarge_size

        return MinMaxBoundingBox(min=xyz_min, max=xyz_max)

    @staticmethod
    def get_point_getter_fn(
        cameras: Cameras, vertices: torch.Tensor, centers: torch.Tensor
    ) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        vertices: [N_partitions * 8, 3], inverse transformed vertices, align with cameras that parsed from origin data
        centers: [N_partitions, 3], bbox centers in partitioning coordinates

        1. Transform vertices to camera coordinates.
        2. For each 8 point:
            1. filter out z<0 points;
            2. project to image;
            3. get ConvexHull of projected image points;
            4. calculate intersection of Polygon(ConvexHull) and image square.
            5. get vertices of intersection in image coordinates as points2d.
            6. adjust points3d dimension (same as points2d, maybe append one vertex of the cube)
        """

        def project(points3D: torch.Tensor, camera: Camera):

            # w2c = torch.eye(4).to(r)
            # w2c[:3, :3] = r
            # w2c[:3, -1] = t
            # k = camera.get_K()[:3, :3]
            # points3D_transformed = points3D @ camera.R.T + camera.T

            w2c_t = camera.world_to_camera
            K_t = camera.get_K().T
            points3D_transformed = torch.cat([points3D, points3D.new_ones((len(points3D), 1))], dim=1) @ w2c_t
            minus_z_mask = points3D_transformed[:, 2] > 0

            # points2D = points3D_transformed @ k.T
            points2D = points3D_transformed @ K_t
            # points2D = torch.cat([points3D, points3D.new_ones((len(points3D), 1))], dim=1) @ camera.get_full_perspective_projection()
            points2D = points2D[:, :2] / points2D[:, 2:3]
            return points2D, minus_z_mask

        def point_getter(camera_idx):
            camera = cameras[camera_idx]
            width, height = camera.width, camera.height
            projected_vertices, minus_z_mask = project(vertices, camera)

            num_points_per_partition = 8
            points_2d, points_3d = vertices.new_empty((0, 2)), vertices.new_empty((0, 3))
            areas = []
            for partition_id, st in enumerate(range(0, len(vertices), num_points_per_partition)):
                _mask = minus_z_mask[st : st + num_points_per_partition]
                if _mask.sum() < 3:
                    areas.append(0.0)
                    continue

                # bug:
                # the input points3D are inverse transformed vertices
                # the center bound_center won't fall into the vertices
                bound_center = centers[partition_id]

                partition_vertices_projected = projected_vertices[st : st + num_points_per_partition][_mask]
                normalized_partition_bound = partition_vertices_projected.cpu().numpy() / np.array([[width, height]])
                convex_hull = ConvexHull(normalized_partition_bound)
                convex_hull_vertices = []
                for v in convex_hull.vertices:
                    convex_hull_vertices.append(normalized_partition_bound[v])
                convex_hull_polygon = Polygon(convex_hull_vertices)
                image_bounds = box(0, 0, 1, 1)
                intersection: Polygon = convex_hull_polygon.intersection(image_bounds)

                areas.append(intersection.area)
                if not intersection.is_empty:
                    intersection_vertices = torch.tensor(list(intersection.exterior.coords))
                    points_2d = torch.cat([points_2d, intersection_vertices], 0)
                    points_3d = torch.cat([points_3d, bound_center.repeat(len(intersection_vertices), 1)], 0)

            projected_points = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0]).reshape(4, 2).to(vertices)

            return (points_2d, points_3d, projected_points)

        return point_getter
