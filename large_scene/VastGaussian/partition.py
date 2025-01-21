import json
import os
import os.path as osp
import pickle
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import torch
import yaml
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.dataparser import ImageSet
from utils.partitioning_utils import VastGSScene, VastGSSceneConfig


@dataclass
class VastGSPartitiongConfig:
    dataset_path: str
    output_path: str
    manhattan_trans: str = None
    partition_dim: List = None

    min_track_length: int = 3
    max_error: float = 2.0
    scene_bbox_enlarge_by_camera_bbox: float = 0.2
    location_based_enlarge: float = 0.2
    visibility_based_partition_enlarge: float = 0.0
    visibility_threshold: float = 0.25

    @staticmethod
    def configure_argparser(parser: ArgumentParser):
        parser.add_argument(
            "--dataset_path",
            required=True,
            type=str,
            default="",
            help="Path to dataset. Containing directories images, sparse, etc.",
        )
        parser.add_argument("--output_path", required=True, type=str, default="", help="Partition info dir.")
        parser.add_argument(
            "--partition_dim",
            required=True,
            type=str,
            default="3,3",
            help="Split number along x- and z-axis, like '2,4'",
        )
        parser.add_argument(
            "--manhattan_trans", type=str, default="manhattan.txt", help="Relative path to dataset_path"
        )
        parser.add_argument("--min_track_length", type=int, default=3)
        parser.add_argument("--max_error", type=float, default=2.0)
        parser.add_argument("--scene_bbox_enlarge_by_camera_bbox", type=float, default=0.2)
        parser.add_argument("--location_based_enlarge", type=float, default=0.2)
        parser.add_argument("--visibility_based_partition_enlarge", type=float, default=0.0)
        parser.add_argument("--visibility_threshold", type=float, default=0.25)

        return parser

    @classmethod
    def instantiate(cls, parser: ArgumentParser):
        args = parser.parse_args()
        partition_dim = [int(s) for s in args.partition_dim.split(",")]
        if len(partition_dim) < 3:
            partition_dim += [1]
        assert len(partition_dim) == 3
        args.partition_dim = partition_dim

        seq = [1.0 if i == j else 0.0 for j in range(4) for i in range(4)]
        if len(args.manhattan_trans) > 0:
            manhattan_path = osp.join(args.dataset_path, args.manhattan_trans)
            if osp.exists(manhattan_path):
                try:
                    with open(manhattan_path, "r") as f:
                        trans_mat = " ".join([l.strip() for l in f.readlines()])
                    seq = [float(s) for s in trans_mat.split()]
                    assert len(seq) == 16, "Invalid manhattan transformation."
                except:
                    print("Parse manhattan transformation failed.")
                    pass
        args.manhattan_trans = ",".join([str(s) for s in seq])

        return cls(**vars(args))


class VastGSPartitioning:
    def __init__(self, config: VastGSPartitiongConfig):
        self.config = config
        self.dataset_path = config.dataset_path
        self.manhattan_trans = self.load_manhattan_transformation(config.manhattan_trans)
        scene_config = VastGSSceneConfig(
            partition_dim=torch.tensor(self.config.partition_dim),
            scene_bbox_enlarge_by_camera_bbox=self.config.scene_bbox_enlarge_by_camera_bbox,
            location_based_enlarge=self.config.location_based_enlarge,
            visibility_based_partition_enlarge=self.config.visibility_based_partition_enlarge,
            visibility_threshold=self.config.visibility_threshold,
        )
        self.scene = VastGSScene(scene_config=scene_config)
        self.output_path = self.config.output_path
        os.makedirs(self.output_path, exist_ok=True)
        yaml.safe_dump(asdict(self.config), open(osp.join(self.output_path, "partition_config.yaml"), "w"))

    @staticmethod
    def load_manhattan_transformation(manhattan_str: str):
        return torch.tensor([float(s) for s in manhattan_str.split(",")]).reshape(4, 4).float()

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

    def read_scene(self):
        dataparser_config = Colmap(split_mode="reconstruction", points_from="random")
        dataparser: ColmapDataParser = dataparser_config.instantiate(
            path=self.dataset_path, output_path=os.getcwd(), global_rank=0
        )
        points3D_path = osp.join(dataparser.detect_sparse_model_dir(), "points3D.bin")
        dataparser_outputs = dataparser.get_outputs()

        image_set = dataparser_outputs.train_set
        # cameras = dataparser_outputs.train_set.cameras
        # image_names = dataparser_outputs.train_set.image_names

        xyzs, rgbs, errors, track_lengths = self.load_points_from_bin(points3D_path)
        mask = self.prefilter_points3D(errors, track_lengths)

        return image_set, [xyzs[mask], rgbs[mask]]

    def prefilter_points3D(self, errors, track_lengths):
        return torch.logical_and(
            torch.ge(track_lengths, self.config.min_track_length),
            torch.le(errors, self.config.max_error),
        )

    def save_plots(self, xyz, rgb):
        os.makedirs(osp.join(self.output_path, "figs"), exist_ok=True)

        # plot scene_bounding_box
        fig, ax = plt.subplots()
        ax.scatter(xyz[::16, 0], xyz[::16, 1], c=rgb[::16] / 255.0, s=1)
        self.scene.plot_scene_bounding_box(ax)
        fig.savefig(osp.join(self.output_path, "figs", "scene_bounding_box.png"), dpi=600)

        self.scene.plot_partitions(ax)
        fig.savefig(osp.join(self.output_path, "figs", "partition_coordinates.png"), dpi=600)
        plt.close(fig)

        coordinates = self.scene.partition_coordinates
        for partition_idx in range(len(coordinates)):
            self.scene.save_plot(
                func=self.scene.plot_partition_assigned_cameras,
                path=osp.join(
                    self.output_path, "figs", "{}-partition.png".format(coordinates.get_str_id(partition_idx))
                ),
                partition_idx=partition_idx,
                point_xyzs=xyz,
                point_rgbs=rgb,
            )

    def save_partitioning_results(self, image_set: ImageSet):
        self.scene.save(
            self.output_path,
            extra_data={
                "up": torch.linalg.inv(self.manhattan_trans)[:3, 1],
                "rotation_transform": self.manhattan_trans,
            },
        )

        is_images_assigned_to_partitions = torch.logical_or(
            self.scene.is_camera_in_partition, self.scene.is_partitions_visible_to_cameras
        )
        written_idx_list = []
        for partition_idx in tqdm(
            list(range(is_images_assigned_to_partitions.shape[0])), desc="Saving image lists and cameras"
        ):
            partition_image_indices = is_images_assigned_to_partitions[partition_idx].nonzero().squeeze(-1).tolist()
            partition_id_str = self.scene.partition_coordinates.get_str_id(partition_idx)
            if len(partition_image_indices) == 0:
                continue
            written_idx_list.append(partition_idx)

            camera_list = []
            os.makedirs(osp.join(self.output_path, "partition_infos", partition_id_str), exist_ok=True)
            with open(osp.join(self.output_path, "partition_infos", partition_id_str, "image_list.txt"), "w") as f:
                for image_index in partition_image_indices:
                    f.write(image_set.image_names[image_index])
                    f.write("\n")

                    color = [0, 0, 255]
                    if self.scene.is_partitions_visible_to_cameras[partition_idx][image_index]:
                        color = [255, 0, 0]
                    camera: Camera = image_set.cameras[image_index]
                    c2w = torch.linalg.inv(camera.world_to_camera.T)
                    camera_list.append(
                        {
                            "id": image_index,
                            "img_name": image_set.image_names[image_index],
                            "width": int(camera.width),
                            "height": int(camera.height),
                            "position": c2w[:3, -1].numpy().tolist(),
                            "rotation": c2w[:3, :3].numpy().tolist(),
                            "fx": float(camera.fx),
                            "fy": float(camera.fy),
                            "cx": camera.cx.item(),
                            "cy": camera.cy.item(),
                            "time": camera.time.item() if camera.item is not None else None,
                            "appearance_id": camera.appearance_id.item() if camera.appearance_id is not None else None,
                            "normalized_appearance_id": (
                                camera.normalized_appearance_id.item()
                                if camera.normalized_appearance_id is not None
                                else None
                            ),
                        }
                    )
            with open(os.path.join(self.output_path, "partition_infos", partition_id_str, "cameras.json"), "w") as f:
                json.dump(camera_list, f, indent=4, ensure_ascii=False)

    def partition(self):
        image_set, (xyzs, rgbs) = self.read_scene()
        reoriented_camera_centers: torch.Tensor = (
            image_set.cameras.camera_center @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        )
        reoriented_points3D: torch.Tensor = xyzs @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        self.scene.camera_centers = reoriented_camera_centers[..., :2]

        # get bounding boxes
        self.scene.get_bounding_box_by_points(reoriented_points3D)
        self.scene.get_bounding_box_by_camera_centers()
        self.scene.get_scene_bounding_box()
        self.scene.build_partition_coordinates()

        # assign cameras to partitions
        self.scene.camera_center_based_partition_assignment()
        vertices = self.scene.get_partition_cube_vertices(reoriented_points3D)
        bbox_centers = vertices.reshape(-1, 8, 3).mean(dim=1)
        inversed_manhattan_trans = torch.linalg.inv(self.manhattan_trans)
        vertices_inverse_manhattan_transformed = (
            vertices @ inversed_manhattan_trans[:3, :3].T + inversed_manhattan_trans[:3, -1]
        )
        self.scene.calculate_camera_visibilities(
            point_getter=self.scene.get_point_getter_fn(
                image_set.cameras,
                vertices_inverse_manhattan_transformed,
                bbox_centers,
            ),
            device=image_set.cameras.R.device,
        )
        self.scene.visibility_based_partition_assignment()

        self.save_plots(reoriented_points3D, rgbs)
        self.save_partitioning_results(image_set)

    @classmethod
    def start(cls, parser, config_cls=VastGSPartitiongConfig):
        config = config_cls.instantiate(parser)
        partitioning = cls(config)
        partitioning.partition()


if __name__ == "__main__":
    parser = ArgumentParser()
    VastGSPartitiongConfig.configure_argparser(parser)
    VastGSPartitioning.start(parser, config_cls=VastGSPartitiongConfig)
