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
from gsplat.rasterize_to_weights import rasterize_to_weights
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
from myimpl.models.implicit_grid_gaussian import (ImplicitGridGaussianModel,
                                                  ImplicitLoDGridGaussianModel)
from myimpl.renderers.grid_renderer import GridGaussianRendererModule

from ..base.partitionable_scene import (PartitionableScene,
                                        PartitionableSceneConfig)


@dataclass
class GridSceneConfig(PartitionableSceneConfig):
    down_sample_factor: int = 4

    partition_bbox_enlarge_by_gaussian_assignment: float = 0.05

    visibility_threshold: float = field(default=0.05)

    train_config: Optional[str] = None

    def instantiate(self):
        return GridScene(self)

    def __post_init__(self):
        self.visibility_threshold = 0.05


@dataclass
class GridScene(PartitionableScene):
    scene_config: GridSceneConfig = field(default_factory=lambda: GridSceneConfig())

    gaussians_in_partitions: torch.Tensor = field(init=False)

    def __post_init__(self):
        super().__post_init__()

    def partition(self, dataset_path: str, output_path: str):
        device = torch.device("cuda")
        # coarse training
        coarse_model, renderer, image_set, ckpt = self.load_coarse_model(dataset_path, output_path, device)

        camera_centers = image_set.cameras.camera_center
        camera_centers_transformed = camera_centers @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        self.camera_centers = camera_centers_transformed[:, :2]

        means = coarse_model.get_xyz.detach().clone().cpu()
        means_transformed = means @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]

        # get bounding boxes
        self.get_bounding_box_by_points(means_transformed)
        self.get_bounding_box_by_camera_centers()
        # get scene bbox by enlarged camera centers
        self.get_scene_bounding_box()
        self.build_partition_coordinates()
        self.camera_center_based_partition_assignment()

        bg_color = ckpt["hyper_parameters"]["background_color"]
        self.gaussians_in_partitions = Partitioning.is_in_bounding_boxes(
            self.partition_coordinates.get_bounding_boxes(
                self.scene_config.partition_bbox_enlarge_by_gaussian_assignment
            ),
            means_transformed[..., :2],
        )
        self.calculate_camera_visibilities(coarse_model, renderer, image_set.cameras, device=device, bg_color=bg_color)
        self.visibility_based_partition_assignment()

        # visualization
        point_cloud = self.load_point_cloud(dataset_path)
        points3D_transformed = (
            point_cloud.points.to(self.manhattan_trans) @ self.manhattan_trans[:3, :3].T + self.manhattan_trans[:3, -1]
        )
        self.save_plots(
            output_path, BasicPointCloud(points=points3D_transformed, colors=point_cloud.colors, normals=None)
        )
        self.save_partitioning_results(output_path, image_set, coarse_model)

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
            if self.scene_config.train_config is not None:
                args += ["--config={}".format(self.scene_config.train_config)]
            else:
                args += [
                    "--data.path={}".format(dataset_path),
                    "--data.parser=Colmap",
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
        ]
        print(" ".join(args))
        subprocess.run(args)

    def load_imageset(self, data_params: Dict[str, Any]):
        dataset_path = data_params["path"]
        dataparser_config = data_params["parser"]
        dataparser_config.points_from = "random"
        dataparser_config.split_mode = "reconstruction"
        dataparser: ColmapDataParser = dataparser_config.instantiate(
            path=dataset_path, output_path=os.getcwd(), global_rank=0
        )
        dataparser_outputs = dataparser.get_outputs()
        return dataparser_outputs.train_set

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

        self.scene_bounding_box = SceneBoundingBox(
            bounding_box=scene_bbox,
            n_partitions=self.partition_dim[:2],
            origin_partition_offset=None,
        )
        return self.scene_bounding_box

    def build_partition_coordinates(self):
        ids, xys = torch.empty([0, 2], dtype=torch.int), torch.empty([0, 2], dtype=torch.float32)
        scene_bbox = self.scene_bounding_box.bounding_box
        sizes = (scene_bbox.max - scene_bbox.min)[:2] / self.partition_dim[:2]
        origin = scene_bbox.min[:2]

        for i in range(self.partition_dim[0]):
            for j in range(self.partition_dim[1]):
                xy = origin + torch.tensor([i, j], dtype=torch.float32) * sizes
                ids = torch.cat([ids, torch.tensor([[i, j]], dtype=torch.int)])
                xys = torch.cat([xys, xy.view(1, 2)])
        sizes = sizes.repeat(len(ids), 1)
        self.partition_coordinates = PartitionCoordinates(id=ids, xy=xys, size=sizes)
        return self.partition_coordinates

    def calculate_camera_visibilities(
        self,
        coarse_model: ImplicitLoDGridGaussianModel,
        renderer: GridGaussianRendererModule,
        cameras: Cameras,
        device="cpu",
        bg_color=...,
    ):
        self.camera_visibilities = torch.zeros(
            (len(self.partition_coordinates), len(cameras)),
            dtype=torch.float32,
        )
        for camera_idx, camera in enumerate(tqdm(cameras)):
            camera.to_device(device)
            output_pkg = renderer(camera, coarse_model, bg_color, render_types=[])
            means2d, conics, opacities, isects, anchor_mask, primitive_mask = (
                output_pkg["viewspace_points"],
                output_pkg["conics"],
                output_pkg["opacities"],
                output_pkg["isects"],
                output_pkg["anchor_mask"],
                output_pkg["primitive_mask"],
            )
            _, _, flatten_ids, isect_offsets = isects
            image_width, image_height = int(camera.width.item()), int(camera.height.item())
            _, _, blend_weights, _ = rasterize_to_weights(
                means2d=means2d.unsqueeze(0),
                conics=conics,
                opacities=opacities.unsqueeze(0).contiguous(),
                image_width=image_width,
                image_height=image_height,
                tile_size=renderer.config.block_size,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                pixel_weights=means2d.new_zeros((1, image_height, image_width)),
            )

            for partition_idx in range(len(self.partition_coordinates)):
                is_in_partition = self.gaussians_in_partitions[partition_idx].to(device)

                primitive_weights = blend_weights.new_zeros((anchor_mask.sum() * coarse_model.n_offsets,))
                primitive_weights[primitive_mask] = blend_weights
                anchor_primitive_weights = blend_weights.new_zeros((anchor_mask.shape[0], coarse_model.n_offsets))
                anchor_primitive_weights[anchor_mask] = primitive_weights.reshape(-1, coarse_model.n_offsets)
                anchor_weights = anchor_primitive_weights.sum(-1)

                weights_in_partition = anchor_weights[is_in_partition].sum()
                weights_total = blend_weights.sum()
                self.camera_visibilities[partition_idx, camera_idx] = weights_in_partition / weights_total

        return self.camera_visibilities

    def visibility_based_partition_assignment(self):
        self.is_partitions_visible_to_cameras = self.camera_visibilities > self.scene_config.visibility_threshold
        return self.is_partitions_visible_to_cameras

    def load_point_cloud(self, dataset_path):
        from glob import glob

        from internal.utils.graphics_utils import (
            fetch_ply_without_rgb_normalization, store_ply)

        ply_path = osp.join(dataset_path, "point_cloud.ply")
        if osp.exists(ply_path):
            point_cloud = fetch_ply_without_rgb_normalization(ply_path)
            points, colors = torch.from_numpy(point_cloud.points), torch.from_numpy(point_cloud.colors)
        else:
            points3D_path = osp.join(dataset_path, "sparse", "**", "points3D.bin")
            points3D_path = list(glob(points3D_path, recursive=True))
            assert len(points3D_path) > 0, "points3D.bin not found"
            points, colors, _, _ = self.load_points_from_bin(points3D_path[0])
        if not osp.exists(ply_path):
            store_ply(ply_path, points, colors)
        return BasicPointCloud(points=points, colors=colors, normals=None)

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

    def save_partitioning_results(self, output_path: str, image_set: ImageSet, model: ImplicitLoDGridGaussianModel):
        super().save_partitioning_results(output_path, image_set)

        partition_dir = osp.join(output_path, "partition_infos")
        for d in os.listdir(osp.join(partition_dir, "partitions")):
            fulldir = osp.join(partition_dir, "partitions", d)
            if osp.isdir(fulldir):
                shutil.copy(osp.join(output_path, "coarse", "cfg_args"), osp.join(fulldir, "cfg_args"))

        model_path = self.save_gaussians(osp.join(output_path, "partition_infos"), model)
        for partition_idx in tqdm(range(len(self.partition_coordinates)), desc="Saving partition ply files"):
            partition_id_str = self.partition_coordinates.get_str_id(partition_idx)
            os.symlink(
                osp.abspath(model_path),
                osp.abspath(osp.join(partition_dir, "partitions", partition_id_str, osp.basename(model_path))),
            )

    def save_gaussians(self, dst_dir: str, model: VanillaGaussianModel):
        if isinstance(model, VanillaGaussianModel):
            dst_path = osp.join(dst_dir, "gaussian_model.ply")
            GaussianPlyUtils.load_from_model(model).to_ply_format().save_to_ply(dst_path)
        elif hasattr(model, "get_anchors"):
            from myimpl.utils.grid_gaussian_loader import GridGaussianUtils

            dst_path = osp.join(dst_dir, "gaussian_model.pt")
            pt = GridGaussianUtils.tensors_from_model(model)
            torch.save(pt, dst_path)
        else:
            raise ValueError("unsupported model type")
        return dst_path
