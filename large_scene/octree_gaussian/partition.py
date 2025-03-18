import json
import os
import os.path as osp
import pickle
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import torch
import yaml
from large_scene.
from large_scene..partition_utils import (OctreeScene,
                                                         OctreeSceneConfig)
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.dataparser import ImageSet
from internal.utils.partitioning_utils import Partitioning
from myimpl.utils.grid_gaussian_utils import GridGaussianUtils


@dataclass
class GridParitioningConfig:
    dataset_path: str
    output_path: str
    manhattan_trans: str = None
    partition_dim: List = None

    min_track_length: int = 3
    max_error: float = 2.0

    scene_bbox_enlarge_by_camera_bbox: float = 0.2
    location_based_enlarge: float = 0.2
    visibility_threshold: float = 0.0  # add all visible camera

    base_layer: int = 15
    fork: int = 2

    @staticmethod
    def configure_argparser(parser: ArgumentParser):
        parser.add_argument(
            "--dataset_path",
            required=True,
            type=str,
            help="Path to dataset. Containing directories images, sparse, etc.",
        )
        parser.add_argument("--output_path", required=True, type=str, help="Partition info dir.")
        parser.add_argument(
            "--partition_dim",
            required=True,
            type=str,
            help="Split number along x- and z-axis, like '2,4'",
        )
        parser.add_argument("--manhattan_trans", type=str, default=None, help="Relative path to dataset_path")
        parser.add_argument("--min_track_length", type=int, default=3)
        parser.add_argument("--max_error", type=float, default=2.0)

        parser.add_argument("--scene_bbox_enlarge_by_camera_bbox", type=float, default=0.2)
        parser.add_argument("--location_based_enlarge", type=float, default=0.2)
        parser.add_argument("--visibility_threshold", type=float, default=0.25)


class GridPartitioning:
    def __init__(self, config: GridParitioningConfig):
        self.config = config
        self.dataset_path = config.dataset_path
        self.manhattan_trans = self.load_manhattan_transformation(config.manhattan_trans)
        self.output_path = self.config.output_path
        scene_config = OctreeSceneConfig(
            partition_dim=torch.tensor(self.config.partition_dim),
            scene_bbox_enlarge_by_camera_bbox=self.config.scene_bbox_enlarge_by_camera_bbox,
            location_based_enlarge=self.config.location_based_enlarge,
            visibility_threshold=self.config.visibility_threshold,
        )
        self.scene = OctreeScene(scene_config=scene_config)
        self.output_path = self.config.output_path
        os.makedirs(osp.join(self.output_path, "partition_infos"), exist_ok=True)
        yaml.safe_dump(
            asdict(self.config), open(osp.join(self.output_path, "partition_infos", "partition_config.yaml"), "w")
        )

    @staticmethod
    def load_manhattan_transformation(manhattan_path: str):
        mat = torch.eye(4, dtype=torch.float)
        if len(manhattan_path) > 0 and osp.exists(manhattan_path):
            try:
                with open(manhattan_path, "r") as f:
                    trans_mat = " ".join([l.strip() for l in f.readlines()])
                mat = torch.tensor([float(s) for s in trans_mat.split()]).reshape(4, 4).float()
            except:
                print("Parse manhattan transformation failed.")
                pass
        return mat

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

    def prefilter_points3D(self, errors, track_lengths):
        return torch.logical_and(
            torch.ge(track_lengths, self.config.min_track_length),
            torch.le(errors, self.config.max_error),
        )

    def read_scene(self):
        dataparser_config = Colmap(split_mode="reconstruction", points_from="random")
        dataparser: ColmapDataParser = dataparser_config.instantiate(
            path=self.dataset_path, output_path=os.getcwd(), global_rank=0
        )
        points3D_path = osp.join(dataparser.detect_sparse_model_dir(), "points3D.bin")
        dataparser_outputs = dataparser.get_outputs()

        image_set = dataparser_outputs.train_set

        xyzs, rgbs, errors, track_lengths = self.load_points_from_bin(points3D_path)
        mask = self.prefilter_points3D(errors, track_lengths)

        return image_set, [xyzs[mask], rgbs[mask]]

    def build_lod_grid(self, xyz, camera_centers: torch.Tensor):
        camera_infos = torch.cat([camera_centers, camera_centers.new_ones((camera_centers.shape[0], 1))], dim=-1)
        self.standard_dist, self.max_level = GridGaussianUtils.get_levels_by_distances(
            xyz, camera_infos, fork=self.config.fork
        )
        self.voxel_size, self.grid_origin = GridGaussianUtils.build_multi_level_grid(
            xyz,
            self.config.scene_bbox_enlarge_by_camera_bbox,
            self.config.base_layer,
            self.config.fork,
            max_level=self.max_level,
        )
        positions, levels = GridGaussianUtils.multi_level_voxelize(
            xyz, self.voxel_size, self.max_level, xyz2grid=self.xyz2grid, grid2xyz=self.grid2xyz
        )
        mask = GridGaussianUtils.weed_out_mask_by_level(
            positions,
            levels,
            0.0,
            cam_infos=camera_infos,
            predict_level_fn=self.map_to_int_level,
            int_level_fn=lambda x: self.map_to_int_level(x, self.max_level)[0],
        )
        self.visibility_thresh = torch.mean(mask.float())
        return positions, levels

    def partition(self):
        image_set, (xyz, rgb) = self.read_scene()
        camera_centers = image_set.cameras.camera_center

        reoriented_camera_centers: torch.Tensor = (
            camera_centers @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        )
        reoriented_xyz = xyz @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        self.scene.camera_centers = reoriented_camera_centers[..., :2]

        self.scene.get_bounding_box_by_points(reoriented_xyz)
        self.scene.get_bounding_box_by_camera_centers()
        self.scene.get_scene_bounding_box()
        self.scene.build_partition_coordinates()

        # sample anchors
        positions, levels = self.build_lod_grid(reoriented_xyz, reoriented_camera_centers)  # reoriented positions

        # assign anchors to partitions
        shared_state_dict = {
            "_voxel_size": self.voxel_size,
            "_grid_origin": self.grid_origin,
            "_max_level": self.max_level,
            "_start_level": (self.max_level / 2).int(),
            "_standard_dist": self.standard_dist,
            "_visibility_threshold": self.visibility_thresh,
        }
        partition_state_dict = {}
        for index, bbox in self.scene.partition_coordinates.get_bounding_boxes(
            enlarge=self.config.location_based_enlarge
        ):
            partition_id_str = self.scene.partition_coordinates.get_str_id(index)
            mask = Partitioning.is_in_bounding_boxes(bbox, positions[..., :2])
            _anchors, _levels = xyz[mask], levels[mask]
            partition_state_dict[partition_id_str] = {"means": _anchors, "levels": _levels}

        self.scene.camera_center_based_partition_assignment()  # assign cameras according to camera positions
        self.scene.calculate_camera_visibilities()

    def xyz2grid(self, points: torch.Tensor, voxel_size: float):
        return torch.round((points - self.grid_origin.to(points)) / voxel_size).int()

    def grid2xyz(self, grid: torch.Tensor, voxel_size: float):
        return grid.float() * voxel_size + self.grid_origin.to(grid.device)

    def predict_level(self, dists: torch.Tensor):
        return GridGaussianUtils.predict_level(dists, standard_dist=self.standard_dist, fork=self.config.fork)

    def map_to_int_level(self, pred_level: torch.Tensor, cur_level: int):
        return GridGaussianUtils.map_to_int_level(pred_level, cur_level, dist2level="round")
