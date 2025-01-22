import json
import os
import os.path as osp
import pickle
import shutil
import subprocess
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from matplotlib import pyplot as plt
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
from utils.partitioning_utils import CityGSScene, CityGSSceneConfig


@dataclass
class CityGSPartitiongConfig:
    dataset_path: str = None
    output_path: str = None
    manhattan_trans: str = None
    partition_dim: List[int] = None
    down_sample_factor: int = 4
    num_gaussians_per_partition_threshold: int = 25_000
    gaussian_bbox_enlarge_step: List = None
    location_based_enlarge: float = 0.2
    visibility_threshold: float = 0.08

    @staticmethod
    def configure_argparser(parser: ArgumentParser):
        parser.add_argument(
            "--dataset_path",
            required=True,
            type=str,
            default="datasets/MegaNeRF/rubble/colmap",
            help="Path to dataset. Containing directories images, sparse, etc.",
        )
        parser.add_argument(
            "--output_path",
            required=True,
            type=str,
            default="tmp/citygs/rubble-2_2",
            help="Partition info dir.",
        )
        parser.add_argument(
            "--partition_dim",
            required=True,
            type=str,
            default="2,2",
            help="Split number along x- and z-axis, like '2,4'",
        )
        parser.add_argument(
            "--manhattan_trans",
            type=str,
            default="manhattan.txt",
            help="Relative path to dataset_path",
        )
        parser.add_argument("--down_sample_factor", type=int, default=4)
        parser.add_argument("--num_gaussians_per_partition_threshold", type=int, default=25_000)
        parser.add_argument("--gaussian_bbox_enlarge_step", type=str, default="0.02,0.02,0.02")
        parser.add_argument("--location_based_enlarge", type=float, default=0.2)
        parser.add_argument("--visibility_threshold", type=float, default=0.08)
        return parser

    @classmethod
    def instantiate(cls, parser: ArgumentParser):
        args = parser.parse_args()
        partition_dim = [int(s) for s in args.partition_dim.split(",")]
        if len(partition_dim) < 3:
            partition_dim += [1]
        assert len(partition_dim) == 3
        args.partition_dim = partition_dim
        gaussian_bbox_enlarge_step = [float(s) for s in args.gaussian_bbox_enlarge_step.split(",")]
        if len(gaussian_bbox_enlarge_step) < 3:
            gaussian_bbox_enlarge_step += [0.02]
        assert len(gaussian_bbox_enlarge_step) == 3
        args.gaussian_bbox_enlarge_step = gaussian_bbox_enlarge_step

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


class CityGSPartitioning:
    def __init__(self, config: CityGSPartitiongConfig):
        self.config = config
        self.manhattan_trans = self.load_manhattan_transformation(config.manhattan_trans)
        scene_config = CityGSSceneConfig(
            location_based_enlarge=self.config.location_based_enlarge,
            visibility_threshold=self.config.visibility_threshold,
            partition_dim=torch.tensor(self.config.partition_dim),
            num_gaussians_per_partition_threshold=self.config.num_gaussians_per_partition_threshold,
            gaussian_bbox_enlarge_step=torch.tensor(self.config.gaussian_bbox_enlarge_step),
        )
        self.scene = CityGSScene(scene_config=scene_config)
        self.output_path = self.config.output_path
        os.makedirs(self.output_path, exist_ok=True)
        yaml.safe_dump(asdict(self.config), open(osp.join(self.output_path, "partition_config.yaml"), "w"))

    @staticmethod
    def load_manhattan_transformation(manhattan_str: str):
        return torch.tensor([float(s) for s in manhattan_str.split(",")]).reshape(4, 4).float()

    def coarse_train(self):
        args = [
            "python",
            "main.py",
            "fit",
            "--project=coarse",
            "--output={}".format(self.config.output_path),
            "-n=coarse",
            "--data.path={}".format(self.config.dataset_path),
            "--data.parser=Colmap",
        ]
        if next((Path(self.config.output_path) / "coarse").rglob("*.ckpt"), None) is not None:
            ckpt_path = GaussianModelLoader.search_load_file(osp.join(self.config.output_path, "coarse"))
            config_path = next((Path(self.config.output_path) / "coarse").rglob("config.yaml"), None)
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
                "--data.train_max_num_images_to_cache=200",
                "--data.async_caching=true",
                "--logger=tensorboard",
            ]
        print(" ".join(args))
        subprocess.run(args)

    def load_imageset(self, data_params: Dict[str, Any]):
        dataset_path = data_params["path"]
        dataparser_config = data_params["parser"]
        dataparser_config.points_from = "random"
        dataparser: ColmapDataParser = dataparser_config.instantiate(
            path=dataset_path, output_path=os.getcwd(), global_rank=0
        )
        dataparser_outputs = dataparser.get_outputs()
        return dataparser_outputs.train_set

    def save_plots(self, xyz: torch.Tensor, rgb: torch.Tensor):
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
                point_sparsify=32,
            )

    def save_partitioning_results(self, model: VanillaGaussianModel, image_set: ImageSet):  # image_names: List[str]
        self.scene.save(
            self.output_path,
            extra_data={
                "up": torch.linalg.inv(self.manhattan_trans)[:3, 1],
                "rotation_transform": self.manhattan_trans,
                "radius_bounding_box": asdict(self.scene.radius_bounding_box),
                "enlarged_gaussian_bounding_boxes": asdict(self.scene.enlarged_gaussian_bounding_boxes),
                "gaussians_in_partitions": self.scene.gaussians_in_partitions,
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
            with open(os.path.join(self.output_path, "partition_infos", partition_id_str, "cameras.json"), "w") as f:
                json.dump(camera_list, f, indent=4, ensure_ascii=False)
            shutil.copy(
                osp.join(self.output_path, "coarse", "cfg_args"),
                osp.join(self.output_path, "partition_infos", partition_id_str, "cfg_args"),
            )

        complete_properties = model.properties
        for partition_idx in tqdm(range(len(self.scene.partition_coordinates)), desc="Saving partition ply files"):
            partition_id_str = self.scene.partition_coordinates.get_str_id(partition_idx)
            incomplete_properties = {
                k: v[self.scene.gaussians_in_partitions[partition_idx]] for k, v in complete_properties.items()
            }
            model.properties = incomplete_properties
            dst_path = osp.join(self.output_path, "partition_infos", partition_id_str, "gaussian_model.ply")
            GaussianPlyUtils.load_from_model(model).to_ply_format().save_to_ply(dst_path)
        model.properties = complete_properties

    def partition(self):
        # coarse training
        self.coarse_train()

        # load coarse model and render
        device = torch.device("cuda")
        ckpt_path = GaussianModelLoader.search_load_file(osp.join(self.config.output_path, "coarse"))
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

        dir_pp = -self.manhattan_trans[2, :3].repeat(means.shape[0], 1)
        shs_view = coarse_model.get_features.transpose(1, 2).view(-1, 3, (coarse_model.max_sh_degree + 1) ** 2)
        rgb = eval_sh(coarse_model.active_sh_degree, shs_view.detach().cpu(), dir_pp)
        rgb = torch.clamp(rgb + 0.5, 0.0, 1.0).detach().cpu().numpy() * 255.0

        # means and cameras are not transformed
        self.scene.camera_centers = camera_centers_transformed
        self.scene.get_radius_box_by_cameras_and_points(means_transformed, image_set.cameras, self.manhattan_trans)
        self.scene.build_partition_coordinates()

        # location based assignment
        self.scene.camera_center_based_partition_assignment()

        # partition gaussian model based on num gaussians
        self.scene.get_enlarged_gaussian_bboxes(means_transformed)

        # render image with one of the partitions removed
        bg_color = ckpt["hyper_parameters"]["background_color"]
        # self.scene.calculate_camera_visibilities(
        #     coarse_model, renderer, image_set.cameras, device=device, bg_color=bg_color
        # )
        self.scene.camera_visibilities = torch.load("tmp/citygs/rubble-3_3/partitions.pt", "cpu")["visibilities"]
        # assign cameras based on visibilities
        self.scene.visibility_based_partition_assignment()

        self.scene.partition_coordinates = PartitionCoordinates(
            id=self.scene.partition_coordinates.id[:, :2],
            xy=self.scene.partition_coordinates.xy[:, :2],
            size=self.scene.partition_coordinates.size[:, :2],
        )
        # self.save_plots(means_transformed, rgb)
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
