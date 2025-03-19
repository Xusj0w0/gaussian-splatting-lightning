import json
import os
import os.path as osp
import pickle
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from jsonargparse import ArgumentParser, set_docstring_parse_options
from matplotlib import pyplot as plt
from partitioning_utils import CityGSScene, CityGSSceneConfig
from tqdm.auto import tqdm

from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.dataparser import DataParser, ImageSet
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.partitioning_utils import (MinMaxBoundingBox,
                                               PartitionCoordinates,
                                               Partitioning)
from internal.utils.sh_utils import eval_sh

set_docstring_parse_options(attribute_docstrings=True)


@dataclass
class CityGSPartitiongConfig:
    name: str
    """ project name, output dir is `outputs/name` """

    dataset_path: str
    """ path to dataset """

    manhattan_path: str = None
    """ path to manhattan transformation text file, containing 4x4 matrix """

    down_sample_factor: int = 4
    """ down sample factor when coarse training """

    min_track_length: int = 3
    """ minimum track length for prefiltering point cloud """

    max_error: float = 2.0
    """ maximum error for prefiltering point cloud """

    scene_config: CityGSSceneConfig = field(default_factory=lambda: CityGSSceneConfig())

    @classmethod
    def configure_argparser(cls, parser: ArgumentParser):
        parser.add_class_arguments(cls, nested_key=None)

        # modify parser
        container = ArgumentParser()
        container.add_argument("-n", "--name", type=str, required=True)
        container.add_argument("-d", "--dataset_path", type=str, required=True)
        container.add_argument("--scene_config.partition_dim", type=int, nargs="+", required=True, default=[])
        container.add_argument("--scene_config.gaussian_bbox_enlarge_step", type=float, nargs=3, default=[0.01, 0.01, 0.01])
        container.add_argument("--scene_config.radius_bounding_box_ratio", type=float, nargs=6, default=[])

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


class CityGSPartitioning:
    config: CityGSPartitiongConfig

    def __init__(self, config: CityGSPartitiongConfig):
        self.config = config
        self.dataset_path = config.dataset_path
        self.manhattan_trans = self.load_manhattan_transformation(config.manhattan_path)
        self.scene = CityGSScene(scene_config=self.config.scene_config)
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

    def coarse_train(self):
        args = [
            "python",
            "main.py",
            "fit",
            "--project=coarse",
            "--output={}".format(self.output_path),
            "-n=coarse",
            "--data.path={}".format(self.config.dataset_path),
            "--data.parser=Colmap",
        ]
        if next((Path(self.output_path) / "coarse").rglob("*.ckpt"), None) is not None:
            ckpt_path = GaussianModelLoader.search_load_file(osp.join(self.output_path, "coarse"))
            config_path = next((Path(self.output_path) / "coarse").rglob("config.yaml"), None)
            if config_path is not None:
                config = yaml.safe_load(open(str(config_path), "r"))
                max_steps = config["trainer"]["max_steps"]
                ckpt_step = int(osp.splitext(osp.basename(ckpt_path))[0].split("step=")[1])
                if ckpt_step >= max_steps:
                    return ckpt_path
            args += [
                "--config={}".format(str(config_path)) if config_path is not None else "",
                "--ckpt_path={}".format(ckpt_path),
            ]
        else:
            args += [
                "--data.parser.down_sample_factor={}".format(self.config.down_sample_factor),
                "--data.parser.split_mode=experiment",
                "--data.parser.eval_image_select_mode=list",
                "--data.parser.eval_list={}".format(osp.join(self.config.dataset_path, "splits/val_images.txt")),
                "--data.async_caching=true",
                "--data.train_max_num_images_to_cache=256",
                "--logger=tensorboard",
            ]
        print(" ".join(args))
        subprocess.run(args)

    def load_imageset(self, data_params: Dict[str, Any]):
        dataset_path = data_params["path"]
        dataparser_config = data_params["parser"]
        dataparser_config.points_from = "random"
        dataparser: ColmapDataParser = dataparser_config.instantiate(path=dataset_path, output_path=os.getcwd(), global_rank=0)
        dataparser_outputs = dataparser.get_outputs()
        return dataparser_outputs.train_set

    def save_plots(self, xyz: torch.Tensor, rgb: torch.Tensor):
        fig_dir = osp.join(self.output_path, "partition_infos", "figs")
        os.makedirs(fig_dir, exist_ok=True)

        # # plot scene_bounding_box
        fig, ax = plt.subplots()
        ax.scatter(xyz[::16, 0], xyz[::16, 1], c=rgb[::16] / 255.0, s=1)
        self.scene.plot_scene_bounding_box(ax)
        fig.savefig(osp.join(fig_dir, "scene_bounding_box.png"), dpi=600)

        # self.scene.plot_partitions(ax)
        # fig.savefig(osp.join(fig_dir, "partition_coordinates.png"), dpi=600)
        # plt.close(fig)

        coordinates = self.scene.partition_coordinates
        for partition_idx in range(len(coordinates)):
            self.scene.save_plot(
                func=self.scene.plot_partition_assigned_cameras,
                path=osp.join(fig_dir, "{}-partition.png".format(coordinates.get_str_id(partition_idx))),
                partition_idx=partition_idx,
                point_xyzs=xyz,
                point_rgbs=rgb,
                point_sparsify=32,
            )

    def save_partitioning_results(self, model: VanillaGaussianModel, image_set: ImageSet):  # image_names: List[str]
        partition_dir = osp.join(self.output_path, "partition_infos")
        os.makedirs(partition_dir, exist_ok=True)
        self.scene.save(
            partition_dir,
            extra_data={
                "up": torch.linalg.inv(self.manhattan_trans)[:3, 1],
                "rotation_transform": self.manhattan_trans,
                "radius_bounding_box": asdict(self.scene.radius_bounding_box),
                "gaussians_in_partitions": self.scene.gaussians_in_partitions,
            },
        )

        is_images_assigned_to_partitions = torch.logical_or(
            self.scene.is_camera_in_partition, self.scene.is_partitions_visible_to_cameras
        )
        written_idx_list = []
        for partition_idx in tqdm(list(range(is_images_assigned_to_partitions.shape[0])), desc="Saving image lists and cameras"):
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
                                camera.normalized_appearance_id.item() if camera.normalized_appearance_id is not None else None
                            ),
                        }
                    )
            with open(os.path.join(partition_dir, "partitions", partition_id_str, "cameras.json"), "w") as f:
                json.dump(camera_list, f, indent=4, ensure_ascii=False)
            shutil.copy(
                osp.join(self.output_path, "coarse", "cfg_args"),
                osp.join(partition_dir, "partitions", partition_id_str, "cfg_args"),
            )

        complete_properties = model.properties
        for partition_idx in tqdm(range(len(self.scene.partition_coordinates)), desc="Saving partition ply files"):
            partition_id_str = self.scene.partition_coordinates.get_str_id(partition_idx)
            incomplete_properties = {k: v[self.scene.gaussians_in_partitions[partition_idx]] for k, v in complete_properties.items()}
            model.properties = incomplete_properties
            dst_path = osp.join(partition_dir, "partitions", partition_id_str, "gaussian_model.ply")
            GaussianPlyUtils.load_from_model(model).to_ply_format().save_to_ply(dst_path)
        model.properties = complete_properties

    def partition(self):
        # coarse training
        self.coarse_train()

        # load coarse model and render
        device = torch.device("cuda")
        ckpt_path = GaussianModelLoader.search_load_file(osp.join(self.output_path, "coarse"))
        coarse_model, renderer, ckpt = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            ckpt_path, device, pre_activate=False
        )
        coarse_model: VanillaGaussianModel

        # load images and loader
        image_set: ImageSet = self.load_imageset(ckpt["datamodule_hyper_parameters"])

        # calculate points' xyz
        camera_centers = image_set.cameras.camera_center
        camera_centers_transformed = camera_centers @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        means = coarse_model.get_xyz.detach().clone().cpu()
        means_transformed = means @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]

        # visualize gaussian point cloud, viewdirs is manhattan coordinates' z axis
        dir_pp = -self.manhattan_trans[2, :3].repeat(means.shape[0], 1)
        shs_view = coarse_model.get_features.transpose(1, 2).view(-1, 3, (coarse_model.max_sh_degree + 1) ** 2)
        rgb = eval_sh(coarse_model.active_sh_degree, shs_view.detach().cpu(), dir_pp)
        rgb = torch.clamp(rgb + 0.5, 0.0, 1.0).detach().cpu().numpy() * 255.0

        # means and cameras are not transformed
        self.scene.camera_centers = camera_centers_transformed
        self.scene.get_scene_bounding_box(means_transformed, image_set.cameras, self.manhattan_trans)
        self.scene.build_partition_coordinates()

        # location based assignment
        self.scene.camera_center_based_partition_assignment()

        # partition gaussian model based on num gaussians
        self.scene.location_based_gaussian_assignment(means_transformed)

        # render image with one of the partitions removed
        bg_color = ckpt["hyper_parameters"]["background_color"]
        # camera_visibilities_path = osp.join(self.output_path, "partition_infos", "camera_visibilities.pkl")
        # if osp.exists(camera_visibilities_path):
        #     self.scene.camera_visibilities = pickle.load(open(camera_visibilities_path, "rb"))
        # else:
        #     self.scene.calculate_camera_visibilities(coarse_model, renderer, image_set.cameras, device=device, bg_color=bg_color)
        #     pickle.dump(self.scene.camera_visibilities, open(camera_visibilities_path, "wb"))
        self.scene.calculate_camera_visibilities(coarse_model, renderer, image_set.cameras, device=device, bg_color=bg_color)
        # assign cameras based on visibilities
        self.scene.visibility_based_partition_assignment()

        self.scene.partition_coordinates = PartitionCoordinates(
            id=self.scene.partition_coordinates.id[:, :2],
            xy=self.scene.partition_coordinates.xy[:, :2],
            size=self.scene.partition_coordinates.size[:, :2],
        )
        self.save_plots(means_transformed, rgb)
        self.save_partitioning_results(coarse_model, image_set)

    @classmethod
    def start(cls, parser, config_cls=CityGSPartitiongConfig):
        config = config_cls.instantiate(parser)
        partitioning = cls(config)
        partitioning.partition()


if __name__ == "__main__":
    parser = ArgumentParser()
    CityGSPartitiongConfig.configure_argparser(parser)
    CityGSPartitioning.start(parser, config_cls=CityGSPartitiongConfig)
