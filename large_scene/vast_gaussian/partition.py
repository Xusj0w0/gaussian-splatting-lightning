import json
import os
import os.path as osp
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import yaml
from jsonargparse import ArgumentParser, set_docstring_parse_options
from matplotlib import pyplot as plt
from partitioning_utils import VastGSScene, VastGSSceneConfig
from tqdm.auto import tqdm

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.dataparser import ImageSet

set_docstring_parse_options(attribute_docstrings=True)


@dataclass
class VastGSPartitiongConfig:
    name: str
    """ project name, output dir is `outputs/name` """

    dataset_path: str
    """ path to dataset """

    manhattan_path: str = None
    """ path to manhattan transformation text file, containing 4x4 matrix """

    min_track_length: int = 3
    """ minimum track length for prefiltering point cloud """

    max_error: float = 2.0
    """ maximum error for prefiltering point cloud """

    scene_config: VastGSSceneConfig = field(default_factory=lambda: VastGSSceneConfig())

    @classmethod
    def configure_argparser(cls, parser: ArgumentParser):
        parser.add_class_arguments(cls, nested_key=None)

        # modify parser
        container = ArgumentParser()
        container.add_argument("-n", "--name", type=str, required=True)
        container.add_argument("-d", "--dataset_path", type=str, required=True)
        container.add_argument("--scene_config.partition_dim", type=int, nargs="+", required=True, default=[])

        def return_default(a, b):
            return b if a is None else a

        action_dict = {action.dest: action for action in container._actions if "help" not in action.dest}
        for i, action in enumerate(parser._actions):
            if action.dest in action_dict:
                for property in ["option_strings", "_typehint", "required", "nargs", "choices", "const"]:
                    setattr(parser._actions[i], property, getattr(action_dict[action.dest], property, None))
                parser._actions[i].help = return_default(action_dict[action.dest].help, action.help)
                parser._actions[i].default = return_default(action_dict[action.dest].default, action.default)


    @classmethod
    def instantiate(cls, parser: ArgumentParser):
        cfg = parser.instantiate_classes(parser.parse_args())
        return cls(**vars(cfg))


class VastGSPartitioning:
    config: VastGSPartitiongConfig

    def __init__(self, config: VastGSPartitiongConfig):
        self.config = config
        self.dataset_path = config.dataset_path
        self.manhattan_trans = self.load_manhattan_transformation(config.manhattan_path)
        self.scene = VastGSScene(scene_config=self.config.scene_config)
        self.output_path = osp.join("outputs", config.name)
        os.makedirs(osp.join(self.output_path, "partition_infos"), exist_ok=True)

    def load_manhattan_transformation(self, manhattan_file: str):
        manhattan_mat = torch.eye(4).float()
        if manhattan_file is None:
            import glob

            manhattan_file = glob.glob(osp.join(self.dataset_path, "**", "manhattan.txt"), recursive=True)
        if len(manhattan_file) == 0:
            return manhattan_mat
        manhattan_file = manhattan_file[0]
        if not osp.exists(manhattan_file):
            return manhattan_mat

        try:
            with open(manhattan_file, "r") as fid:
                manhattan_seq = []
                for line in fid.readlines():
                    manhattan_seq += map(float, line.strip().split())
                assert len(manhattan_seq) == 16, "manhattan transformation should be 4x4 matrix"
                manhattan_mat = torch.tensor(manhattan_seq).reshape(4, 4).float()
        except:
            pass

        return manhattan_mat

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

    def load_scene(self):
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

    def prefilter_points3D(self, errors, track_lengths):
        return torch.logical_and(
            torch.ge(track_lengths, self.config.min_track_length),
            torch.le(errors, self.config.max_error),
        )

    def save_plots(self, xyz, rgb):
        fig_dir = osp.join(self.output_path, "partition_infos", "figs")
        os.makedirs(fig_dir, exist_ok=True)

        # plot scene_bounding_box
        fig, ax = plt.subplots()
        ax.scatter(xyz[::16, 0], xyz[::16, 1], c=rgb[::16] / 255.0, s=1)
        self.scene.plot_scene_bounding_box(ax)
        fig.savefig(osp.join(fig_dir, "scene_bounding_box.png"), dpi=600)

        self.scene.plot_partitions(ax)
        fig.savefig(osp.join(fig_dir, "partition_coordinates.png"), dpi=600)
        plt.close(fig)

        coordinates = self.scene.partition_coordinates
        for partition_idx in range(len(coordinates)):
            self.scene.save_plot(
                func=self.scene.plot_partition_assigned_cameras,
                path=osp.join(fig_dir, "{}-partition.png".format(coordinates.get_str_id(partition_idx))),
                partition_idx=partition_idx,
                point_xyzs=xyz,
                point_rgbs=rgb,
            )

    def save_partitioning_results(self, image_set: ImageSet):
        partition_dir = osp.join(self.output_path, "partition_infos")
        os.makedirs(partition_dir, exist_ok=True)
        self.scene.save(
            partition_dir,
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
            os.makedirs(osp.join(partition_dir, "partitions", partition_id_str), exist_ok=True)
            with open(osp.join(partition_dir, "partitions", partition_id_str, "image_list.txt"), "w") as f:
                for image_index in partition_image_indices:
                    f.write(image_set.image_names[image_index])
                    f.write("\n")

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
                            "time": camera.time.item() if camera.time is not None else None,
                            "appearance_id": camera.appearance_id.item() if camera.appearance_id is not None else None,
                            "normalized_appearance_id": (
                                camera.normalized_appearance_id.item()
                                if camera.normalized_appearance_id is not None
                                else None
                            ),
                        }
                    )
            with open(os.path.join(partition_dir, "partitions", partition_id_str, "cameras.json"), "w") as f:
                json.dump(camera_list, f, indent=4, ensure_ascii=False)

    def partition(self):
        image_set, (xyzs, rgbs) = self.load_scene()
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
    def start(cls, parser: ArgumentParser, config_cls=VastGSPartitiongConfig):
        config = config_cls.instantiate(parser)
        partitioning = cls(config)

        # save yaml config
        # when loading, configure_argparser, then `parser.parse_path('<yaml_path>')`
        parser.save(parser.parse_args(), osp.join(partitioning.output_path, "partition_infos/config.yaml"))

        partitioning.partition()


if __name__ == "__main__":
    parser = ArgumentParser()
    VastGSPartitiongConfig.configure_argparser(parser)
    VastGSPartitioning.start(parser, config_cls=VastGSPartitiongConfig)
