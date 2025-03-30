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
from internal.models.vanilla_gaussian import VanillaGaussian, VanillaGaussianModel
from internal.renderers.renderer import Renderer
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.graphics_utils import BasicPointCloud
from internal.utils.partitioning_utils import (
    MinMaxBoundingBox,
    MinMaxBoundingBoxes,
    PartitionCoordinates,
    Partitioning,
    SceneBoundingBox,
)
from internal.utils.sh_utils import eval_sh

from ..base.partitionable_scene import PartitionableScene, PartitionableSceneConfig


@dataclass
class CitySceneConfig(PartitionableSceneConfig):
    down_sample_factor: int = 4
    """ down sample factor when coarse training """

    config: str = None
    """ path to config file for coarse training """

    num_gaussians_per_partition_threshold: int = 25_000
    """ keep enlarging bounding box if number of gaussians in it is lower than this threshold """

    gaussian_bbox_enlarge_step: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01])
    """ enlarge step of bounding box """

    radius_bounding_box_ratio: List[float] = field(default_factory=lambda: [])
    """ ratios of radius bounding box to camera center bounding box, [xmin, xmax, ymin, ymax, zmin, zmax] """

    visibility_threshold: float = 0.05

    outlier_ratio: float = 0.01

    def instantiate(self):
        return CityScene(self)


@dataclass
class CityScene(PartitionableScene):
    scene_config: CitySceneConfig = field(default_factory=lambda: CitySceneConfig())

    gaussian_bbox_enlarge_step: torch.Tensor = field(init=False)

    radius_bounding_box: MinMaxBoundingBox = field(default=None, init=False)
    """ Need to be saved for subsequent contraction calculation """

    gaussians_in_partitions: torch.Tensor = field(default=None, init=False)
    """ indicate each gaussian in which partition, [N_partition, N_gaussian] """

    def __post_init__(self):
        super().__post_init__()
        self.gaussian_bbox_enlarge_step = torch.tensor(self.scene_config.gaussian_bbox_enlarge_step)

    def partition(self, dataset_path: str, output_path: str):
        device = torch.device("cuda")
        # coarse training
        coarse_model, renderer, image_set, ckpt = self.load_coarse_model(dataset_path, output_path, device)

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
        self.camera_centers = camera_centers_transformed
        self.get_scene_bounding_box(means_transformed, image_set.cameras)
        self.build_partition_coordinates()

        # location based assignment
        self.camera_center_based_partition_assignment()

        # partition gaussian model based on num gaussians
        self.location_based_gaussian_assignment(means_transformed)

        # render image with one of the partitions removed
        bg_color = ckpt["hyper_parameters"]["background_color"]
        self.calculate_camera_visibilities(coarse_model, renderer, image_set.cameras, device=device, bg_color=bg_color)
        # assign cameras based on visibilities
        self.visibility_based_partition_assignment()

        self.partition_coordinates = PartitionCoordinates(
            id=self.partition_coordinates.id[:, :2],
            xy=self.partition_coordinates.xy[:, :2],
            size=self.partition_coordinates.size[:, :2],
        )
        self.save_plots(output_path, BasicPointCloud(points=means_transformed, colors=rgb, normals=None))
        self.save_partitioning_results(output_path, image_set, coarse_model)

    @classmethod
    def is_in_partition(
        cls,
        coordinates: torch.Tensor,
        partition_bbox: MinMaxBoundingBox,
        manhattan_trans: torch.Tensor,
        radius_bbox: MinMaxBoundingBox,
        *args,
        **kwargs,
    ):
        coordinates_trans = coordinates @ manhattan_trans[:3, :3].T + manhattan_trans[:3, -1]
        coordinates_contracted = cls.contract(coordinates_trans, radius_bbox, torch.inf, *args, **kwargs)
        is_ge_min = torch.prod(torch.ge(coordinates_contracted[..., :2], partition_bbox.min), dim=-1)
        is_lt_max = torch.prod(torch.lt(coordinates_contracted[..., :2], partition_bbox.max), dim=-1)
        is_in_partition = torch.logical_and(is_ge_min, is_lt_max)
        return is_in_partition

    def save_partitioning_results(self, output_path: str, image_set: ImageSet, model: VanillaGaussianModel):
        super().save_partitioning_results(output_path, image_set)

        partition_dir = osp.join(output_path, "partition_infos")
        for d in os.listdir(osp.join(partition_dir, "partitions")):
            fulldir = osp.join(partition_dir, "partitions", d)
            if osp.isdir(fulldir):
                shutil.copy(osp.join(output_path, "coarse", "cfg_args"), osp.join(fulldir, "cfg_args"))

        complete_properties = model.properties
        for partition_idx in tqdm(range(len(self.partition_coordinates)), desc="Saving partition ply files"):
            partition_id_str = self.partition_coordinates.get_str_id(partition_idx)
            incomplete_properties = {k: v[self.gaussians_in_partitions[partition_idx]] for k, v in complete_properties.items()}
            model.properties = incomplete_properties
            dst_path = osp.join(partition_dir, "partitions", partition_id_str, "gaussian_model.ply")
            GaussianPlyUtils.load_from_model(model).to_ply_format().save_to_ply(dst_path)
        model.properties = complete_properties

    def get_extra_data(self):
        extra_data = super().get_extra_data()
        extra_data.update(
            {
                "up": torch.linalg.inv(self.manhattan_trans)[:3, 1],
                "rotation_transform": self.manhattan_trans,
                "radius_bounding_box": asdict(self.radius_bounding_box),
                "gaussians_in_partitions": self.gaussians_in_partitions,
            }
        )
        return extra_data

    def coarse_train(self, dataset_path: str, output_path: str):
        args = ["python", "main.py", "fit"]

        if next((Path(output_path) / "coarse").rglob("*.ckpt"), None) is not None:
            ckpt_path = GaussianModelLoader.search_load_file(osp.join(output_path, "coarse"))
            config_path = next((Path(output_path) / "coarse").rglob("config.yaml"), None)
            if config_path is not None:
                config = yaml.safe_load(open(str(config_path), "r"))
                max_steps = config["trainer"]["max_steps"]
                ckpt_step = int(osp.splitext(osp.basename(ckpt_path))[0].split("step=")[1])
                if ckpt_step >= max_steps:  # training finished
                    return
            args += [
                "--config={}".format(str(config_path)) if config_path is not None else "",
                "--ckpt_path={}".format(ckpt_path),
            ]
        else:
            if self.scene_config.config is not None:
                args += ["--config={}".format(self.scene_config.config)]
            args += [
                "--data.parser.down_sample_factor={}".format(self.scene_config.down_sample_factor),
                "--data.parser.split_mode=experiment",
                "--data.parser.eval_image_select_mode=list",
                "--data.parser.eval_list={}".format(osp.join(dataset_path, "splits/val_images.txt")),
                "--data.async_caching=true",
                "--data.train_max_num_images_to_cache=256",
                "--logger=tensorboard",
            ]
        args += [
            "--project=coarse",
            "--output={}".format(output_path),
            "-n=coarse",
            "--data.path={}".format(dataset_path),
            "--data.parser=Colmap",
        ]
        print(" ".join(args))
        subprocess.run(args)

    def load_coarse_model(
        self, dataset_path: str, output_path: str, device: torch.device
    ) -> Tuple[VanillaGaussianModel, Renderer, ImageSet, Dict[str, Any]]:
        self.coarse_train(dataset_path, output_path)

        # load coarse model and render
        ckpt_path = GaussianModelLoader.search_load_file(osp.join(output_path, "coarse"))
        coarse_model, renderer, ckpt = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            ckpt_path, device, pre_activate=False
        )
        # load images and loader
        image_set: ImageSet = self.load_imageset(ckpt["datamodule_hyper_parameters"])
        return coarse_model, renderer, image_set, ckpt

    def load_imageset(self, data_params: Dict[str, Any]):
        dataset_path = data_params["path"]
        dataparser_config = data_params["parser"]
        dataparser_config.points_from = "random"
        dataparser: ColmapDataParser = dataparser_config.instantiate(path=dataset_path, output_path=os.getcwd(), global_rank=0)
        dataparser_outputs = dataparser.get_outputs()
        return dataparser_outputs.train_set

    def plot_scene_bounding_box(self, ax):
        super().plot_scene_bounding_box(ax)

        # fmt: off
        ax.add_artist(mpatches.Rectangle(
            self.radius_bounding_box.min.tolist(),
            self.radius_bounding_box.max[0] - self.radius_bounding_box.min[0],
            self.radius_bounding_box.max[1] - self.radius_bounding_box.min[1],
            fill=False,
            color="red",
            label="radius_bbox",
        ))
        # fmt: on

    def get_scene_bounding_box(self, points: torch.Tensor, cameras: Cameras):
        if isinstance(self.scene_config.radius_bounding_box_ratio, list) and len(self.scene_config.radius_bounding_box_ratio) == 6:
            bounding_box = self.camera_center_based_bounding_box
            if getattr(self, "camera_center_based_bounding_box", None) is None:
                bounding_box = self.get_bounding_box_by_camera_centers()
            radius_bbox_ratios = MinMaxBoundingBox(
                min=torch.tensor(self.scene_config.radius_bounding_box_ratio[0::2]),
                max=torch.tensor(self.scene_config.radius_bounding_box_ratio[1::2]),
            )
            radius_bbox_min = (1.0 - radius_bbox_ratios.min) * bounding_box.min + radius_bbox_ratios.min * bounding_box.max
            radius_bbox_max = (1.0 - radius_bbox_ratios.max) * bounding_box.min + radius_bbox_ratios.max * bounding_box.max
            self.radius_bounding_box = MinMaxBoundingBox(min=radius_bbox_min, max=radius_bbox_max)
        else:
            camera_centers: torch.Tensor = cameras.camera_center  # [N, 3]
            focal_axes: torch.Tensor = cameras.R[:, -1, :]  # [N, 3]
            Ms = torch.eye(3) - focal_axes.unsqueeze(-1) * focal_axes.unsqueeze(-2)
            MtMs = Ms.transpose(-1, -2) @ Ms
            A, b = MtMs.mean(dim=0), torch.bmm(MtMs, camera_centers.unsqueeze(-1)).squeeze(-1).mean(dim=0)
            # solve Ax=b
            focus_point = torch.linalg.inv(A) @ b  # [3]
            focus_point_transformed = self.manhattan_trans[:3, :3] @ focus_point + self.manhattan_trans[:3, -1]

            camera_centers_transformed = camera_centers @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
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
        scene_bbox_min = torch.quantile(points, self.scene_config.outlier_ratio, dim=0)
        scene_bbox_max = torch.quantile(points, 1.0 - self.scene_config.outlier_ratio, dim=0)

        self.scene_config.origin = (scene_bbox_min + scene_bbox_max) / 2
        self.scene_config.partition_size = ((scene_bbox_max - scene_bbox_min) / self.partition_dim)[:2].max().item()
        self.scene_bounding_box = SceneBoundingBox(
            MinMaxBoundingBox(min=scene_bbox_min, max=scene_bbox_max),
            n_partitions=self.partition_dim,
            origin_partition_offset=None,
        )

        return self.scene_bounding_box

    @classmethod
    def contract(
        cls,
        points: torch.Tensor,
        radius_bbox: MinMaxBoundingBox,
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
            points_unnormalized = ((points_uncontracted + 1.0) / 2.0 * (radius_bbox.max - radius_bbox.min)) + radius_bbox.min
            points_unnormalized[equal_to_2] = torch.where(is_positive, torch.inf, -torch.inf)
            return points_unnormalized
        else:
            points_normalized = (
                (points - radius_bbox.min) / (radius_bbox.max - radius_bbox.min)
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
        grid_x, grid_y, grid_z = torch.meshgrid(*[torch.arange(k) for k in self.partition_dim], indexing="ij")
        id_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).long()
        xyz_min = id_tensor.float() / self.partition_dim
        xyz_min = xyz_min * 4 - 2  # [0, 1] -> [-2, 2]
        xyz_max = (id_tensor.float() + 1) / self.partition_dim
        xyz_max = xyz_max * 4 - 2

        self.partition_coordinates = PartitionCoordinates(id=id_tensor, xy=xyz_min, size=xyz_max - xyz_min)
        return self.partition_coordinates

    def location_based_gaussian_assignment(self, points: torch.Tensor):
        """
        input points are transformed. enlarge in contracted coordinates.
        """
        points_contracted = self.contract(points, self.radius_bounding_box, ord=torch.inf)

        bboxes = self.partition_coordinates.get_bounding_boxes()
        # enlarged_min, enlarged_max = [], []
        self.gaussians_in_partitions = torch.zeros((len(self.partition_coordinates), len(points)), dtype=torch.bool)

        for partition_id in range(len(self.partition_coordinates)):
            # bbox is contracted
            bbox: MinMaxBoundingBox = bboxes[partition_id]
            gaussians_in_block = (
                torch.prod(points_contracted > bbox.min, dim=-1) * torch.prod(points_contracted < bbox.max, dim=-1)
            ) > 0

            bbox_enlarged = deepcopy(bbox)
            num_gaussians = gaussians_in_block.sum()
            while num_gaussians < self.scene_config.num_gaussians_per_partition_threshold:
                bbox_enlarged.min -= self.gaussian_bbox_enlarge_step
                bbox_enlarged.max += self.gaussian_bbox_enlarge_step
                gaussians_in_block = (
                    torch.prod(points_contracted > bbox_enlarged.min, dim=-1)
                    * torch.prod(points_contracted < bbox_enlarged.max, dim=-1)
                ) > 0
                num_gaussians = gaussians_in_block.sum()
            self.gaussians_in_partitions[partition_id].copy_(gaussians_in_block)

    def camera_center_based_partition_assignment(self):
        bounding_boxes = self.partition_coordinates.get_bounding_boxes(
            enlarge=self.scene_config.location_based_enlarge,
        )
        self.is_camera_in_partition = Partitioning.is_in_bounding_boxes(
            bounding_boxes=MinMaxBoundingBoxes(min=bounding_boxes.min, max=bounding_boxes.max),
            coordinates=self.contract(self.camera_centers, self.radius_bounding_box, ord=torch.inf),
        )
        return self.is_camera_in_partition

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
