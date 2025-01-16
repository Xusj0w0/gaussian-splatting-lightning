import math
from copy import deepcopy
from dataclasses import dataclass
import os, os.path as osp
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.utils.partitioning_utils import (
    MinMaxBoundingBox,
    MinMaxBoundingBoxes,
    PartitionableScene,
    PartitionCoordinates,
    Partitioning,
    SceneBoundingBox,
    SceneConfig,
)


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
            _ = colmap_utils.read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)

            xyzs.append(xyz)
            rgbs.append(rgb)
            errors.append(error)
            track_lengths.append(torch.tensor(track_length))
    return list(map(lambda x: torch.stack(x, dim=0), [xyzs, rgbs, errors, track_lengths]))


@dataclass
class VastGSSceneConfig(SceneConfig):
    """
    Deprecate origin, partition_size.
    """

    origin: torch.Tensor = None

    partition_size: float = None

    partition_dim: torch.Tensor = torch.tensor([2, 4, 1])  # [3]
    """ block numbers along x-, y-, and z-axis. """


@dataclass
class VastGSScene(PartitionableScene):
    scene_config: VastGSSceneConfig

    def get_scene_bounding_box(self):
        """
        Orgininal implementation use grids to cover the point-based bbox.
        Here, we use the camera-centers-based bbox directly.
        """
        if self.camera_center_based_bounding_box is None:
            self.get_bounding_box_by_camera_centers()
        self.scene_bounding_box = SceneBoundingBox(
            bounding_box=self.camera_center_based_bounding_box,
            n_partitions=self.scene_config.partition_dim[:2],
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
        x_dim, y_dim, z_dim = self.scene_config.partition_dim.long().tolist()

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

                partition_dict[f"{i+1}_{j+1}"] = {
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
        x_dim, y_dim, z_dim = self.scene_config.partition_dim.long().tolist()
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
                bottom_partition_id = f"{i+1}_{j+1}"
                up_partition_id = f"{i+1}_{j+2}"
                # mid_camera_id = partition_dict[bottom_partition_id]["y_mid_camera_id"]
                # y_mid = camera_positions[mid_camera_id, 1]
                y_mid = 0.5 * (bbox_dict[bottom_partition_id].max[1] + bbox_dict[up_partition_id].min[1])
                bbox_dict[bottom_partition_id].max[1] = y_mid
                bbox_dict[up_partition_id].min[1] = y_mid

        # Refine along x-axis
        for j in range(y_dim):
            for i in range(x_dim - 1):
                left_partition_id = f"{i+1}_{j+1}"
                right_partition_id = f"{i+2}_{j+1}"
                # mid_camera_id = partition_dict[left_partition_id]["x_mid_camera_id"]
                # x_mid = camera_positions[mid_camera_id, 0]
                x_mid = 0.5 * (bbox_dict[left_partition_id].max[0] + bbox_dict[right_partition_id].min[0])
                bbox_dict[left_partition_id].max[0] = x_mid
                bbox_dict[right_partition_id].min[0] = x_mid

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
            z_bound = self.get_3D_bounding_box_by_points(points=xyzs_in_partition[..., 2:], outlier_threshold=0.005)
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
        cameras: Cameras, vertices: torch.Tensor
    ) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        vertices: [N_partitions * 8, 3].
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
            w2c_t = camera.world_to_camera
            K_t = camera.get_K().T
            points3D_transformed = torch.cat([points3D, points3D.new_ones((len(points3D), 1))], dim=1) @ w2c_t
            minus_z_mask = points3D_transformed[:, 2] > 0
            points2D = points3D_transformed @ K_t
            # points2D = torch.cat([points3D, points3D.new_ones((len(points3D), 1))], dim=1) @ camera.get_full_perspective_projection()
            points2D = points2D[:, :2] / points2D[:, 2:3]
            return points2D, minus_z_mask

        def point_getter(camera_idx):
            camera = cameras[camera_idx]
            half_width, half_height = camera.width / 2, camera.height / 2
            projected_vertices, minus_z_mask = project(vertices, camera)

            num_points_per_partition = 8
            points_2d, points_3d = vertices.new_empty((0, 2)), vertices.new_empty((0, 3))
            for st in range(0, len(vertices), num_points_per_partition):
                _mask = minus_z_mask[st : st + num_points_per_partition]
                if _mask.sum() < 3:
                    continue
                bound_center = vertices[st : st + num_points_per_partition].mean(dim=0)

                partition_vertices_projected = projected_vertices[st : st + num_points_per_partition][_mask]
                normalized_partition_bound = partition_vertices_projected.cpu().numpy() / np.array(
                    [[half_width, half_height]]
                )
                convex_hull = ConvexHull(normalized_partition_bound)
                convex_hull_vertices = []
                for v in convex_hull.vertices:
                    convex_hull_vertices.append(normalized_partition_bound[v])
                convex_hull_polygon = Polygon(convex_hull_vertices)
                image_bounds = box(-1, -1, 1, 1)
                intersection: Polygon = convex_hull_polygon.intersection(image_bounds)
                if not intersection.is_empty:
                    intersection_vertices = torch.tensor(list(intersection.exterior.coords))
                    points_2d = torch.cat([points_2d, intersection_vertices], 0)
                    points_3d = torch.cat([points_3d, bound_center.repeat(len(intersection_vertices), 1)], 0)

            projected_points = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).to(vertices)

            return (points_2d, points_3d, projected_points)

        return point_getter

    def save_plot(self, func: Callable, path: str, *args, **kwargs):
        """
        `plt.show()` may cause error.
        """
        fig, ax = plt.subplots()
        func(ax, *args, **kwargs)
        plt.savefig(path, dpi=600)
        plt.close(fig)

    def plot(self, *args, **kwargs):
        """
        `plt.show()` may cause error.
        """
        return

    def set_plot_ax_limit(self, ax, plot_enlarge: float = 0.25):
        enlarged_min = self.scene_bounding_box.bounding_box.min - plot_enlarge * (
            self.scene_bounding_box.bounding_box.max - self.scene_bounding_box.bounding_box.min
        )
        enlarged_max = self.scene_bounding_box.bounding_box.max + plot_enlarge * (
            self.scene_bounding_box.bounding_box.max - self.scene_bounding_box.bounding_box.min
        )
        ax.set_xlim([enlarged_min[0], enlarged_max[0]])
        ax.set_ylim([enlarged_min[1], enlarged_max[1]])


if __name__ == "__main__":
    # load and parse data
    dataparser_config = Colmap(split_mode="reconstruction", down_sample_factor=4, points_from="random")
    dataparser: ColmapDataParser = dataparser_config.instantiate(
        path="/data/xusj/Projects/3drec/gaussian-splatting-lightning/datasets/MegaNeRF/rubble/colmap",
        output_path=os.getcwd(),
        global_rank=0,
    )
    points3D_path = osp.join(dataparser.detect_sparse_model_dir(), "points3D.bin")
    dataparser_outputs = dataparser.get_outputs()

    cameras: Cameras = dataparser_outputs.train_set.cameras
    if not osp.exists("tmp.pkl"):
        xyzs, rgbs, errors, track_length = load_points_from_bin(points3D_path)  # need pre-filter
        interv = 16
        pickle.dump([xyzs[::interv], rgbs[::interv], errors[::interv], track_length[::interv]], open("tmp.pkl", "bw"))
    else:
        xyzs, rgbs, errors, track_length = pickle.load(open("tmp.pkl", "br"))
    # points: torch.Tensor = ...  # [N, 3]

    # manhattan align
    manhattan_trans: torch.Tensor = ...  # [4, 4]
    manhattan_trans = torch.tensor(
        [
            [0.924456, 0.000000, 0.381288, 26.049109],
            [-0.381288, 0.000000, 0.924456, 45.106720],
            [0.000000, -1.000000, 0.000000, 29.000000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ]
    )
    inversed_manhattan_trans = torch.linalg.inv(manhattan_trans)
    reoriented_camera_centers: torch.Tensor = (
        cameras.camera_center @ manhattan_trans[:3, :3].T + manhattan_trans[:3, -1]
    )
    reoriented_points: torch.Tensor = xyzs @ manhattan_trans[:3, :3].T + manhattan_trans[:3, -1]

    # create VastGSScene
    output_path = "/data/xusj/Projects/3drec/gaussian-splatting-lightning/tmp/partitions/rubble-debug"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(osp.join(output_path, "figs"), exist_ok=True)

    scene_config = VastGSSceneConfig(
        location_based_enlarge=0.2,
        visibility_based_partition_enlarge=0.0,
        visibility_threshold=0.25,
        convex_hull_based_visibility=True,
        visibility_based_distance=0.0,
    )
    scene = VastGSScene(
        scene_config=scene_config,
        camera_centers=reoriented_camera_centers[..., :2],
    )
    scene.get_bounding_box_by_points(reoriented_points)
    scene.get_bounding_box_by_camera_centers()
    scene.build_partition_coordinates()

    # actually bbox defined by points
    # call get_scene_bounding_box() before call plot functions
    scene.get_scene_bounding_box()
    scene.save_plot(
        func=scene.plot_scene_bounding_box,
        path=osp.join(output_path, "figs", "scene_bbox.png"),
    )  # only bounding box
    scene.save_plot(
        func=scene.plot_partitions,
        path=osp.join(output_path, "figs", "partitions.png"),
    )

    # assign cameras based on partition coordinates
    # scene_config.location_based_enlarge
    scene.camera_center_based_partition_assignment()

    # calculate camera visibilities
    # assign cameras based on visibilities
    vertices = scene.get_partition_cube_vertices(reoriented_points)
    vertices_inverse_manhattan_transformed = (
        vertices @ inversed_manhattan_trans[:3, :3].T + inversed_manhattan_trans[:3, -1]
    )
    scene.calculate_camera_visibilities(
        point_getter=scene.get_point_getter_fn(cameras, vertices_inverse_manhattan_transformed),
        device=cameras.R.device,
    )
    scene.visibility_based_partition_assignment()
    print()
