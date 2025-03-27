import glob
import json
import os
import os.path as osp
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, overload

import numpy as np
import torch
import yaml
from jsonargparse import ArgumentParser
from jsonargparse.typing import final
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers.dataparser import ImageSet
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.graphics_utils import BasicPointCloud
from internal.utils.partitioning_utils import MinMaxBoundingBox
from internal.utils.partitioning_utils import \
    PartitionableScene as _PartitionableScene
from internal.utils.partitioning_utils import SceneConfig as _SceneConfig


class SceneConfigBase:
    """To get class_path and init_args"""

    pass


@dataclass
class PartitionableSceneConfig(SceneConfigBase, _SceneConfig):

    origin: torch.Tensor = field(init=False)

    partition_size: float = field(init=False)

    partition_dim: List[int] = field(default_factory=lambda: [1, 1])

    manhattan_trans: Union[List[float], str] = None

    location_based_enlarge: float = 0.1
    """ enlarge bounding box by `partition_size * location_based_enlarge`, used for location based camera assignment """

    visibility_based_distance: float = 0.0
    """ enlarge bounding box by `partition_size * visibility_based_distance`, used for visibility based camera assignment """

    visibility_based_partition_enlarge: float = 0.2
    """ enlarge bounding box by `partition_size * location_based_enlarge`, the points in this bounding box will be treated as inside partition """

    visibility_threshold: float = 0.25

    convex_hull_based_visibility: bool = True
    """ convex hull based visibility calculation """

    scene_bbox_enlarge_by_camera_bbox: float = 0.2
    """ enlarge scene bounding box by camera bounding box. """

    def __post_init__(self):
        self.get_partition_dim()

    def instantiate(self) -> "PartitionableScene":
        pass

    def get_partition_dim(self):
        assert len(self.partition_dim) in [2, 3], "Only 2D or 3D partition is supported."
        if len(self.partition_dim) == 2:
            partition_dim = self.partition_dim + [1]
        else:
            partition_dim = self.partition_dim
        self.partition_dim = partition_dim


@dataclass
class PartitionableScene(_PartitionableScene):

    partition_dim: torch.Tensor = field(init=False)

    manhattan_trans: torch.Tensor = field(init=False)

    scene_config: PartitionableSceneConfig = field(default_factory=lambda: PartitionableSceneConfig())

    def __post_init__(self):
        self.partition_dim = torch.tensor(self.scene_config.partition_dim)
        try:
            self.manhattan_trans = torch.tensor(self.scene_config.manhattan_trans).reshape(4, 4)
        except:
            self.manhattan_trans = torch.eye(4).float()

    def partition(self, dataset_path: str, output_path: str):
        pass

    @classmethod
    def is_in_partition(
        cls,
        coordinates: torch.Tensor,
        partition_bbox: MinMaxBoundingBox,
        manhattan_trans: torch.Tensor,
        *args,
        **kwargs,
    ):
        coordinates_trans = coordinates @ manhattan_trans[:3, :3].T + manhattan_trans[:3, -1]
        is_ge_min = torch.prod(torch.ge(coordinates_trans[..., :2], partition_bbox.min), dim=-1)
        is_lt_max = torch.prod(torch.lt(coordinates_trans[..., :2], partition_bbox.max), dim=-1)
        is_in_partition = torch.logical_and(is_ge_min, is_lt_max)
        return is_in_partition

    def save_plots(self, output_path: str, point_cloud: BasicPointCloud):
        fig_dir = osp.join(output_path, "partition_infos", "figs")
        os.makedirs(fig_dir, exist_ok=True)

        # plot scene_bounding_box
        fig, ax = plt.subplots()
        ax.scatter(point_cloud.points[::16, 0], point_cloud.points[::16, 1], c=point_cloud.colors[::16] / 255.0, s=1)
        self.plot_scene_bounding_box(ax)
        fig.savefig(osp.join(fig_dir, "scene_bounding_box.png"), dpi=600)

        self.plot_partitions(ax)
        fig.savefig(osp.join(fig_dir, "partition_coordinates.png"), dpi=600)
        plt.close(fig)

        coordinates = self.partition_coordinates
        for partition_idx in range(len(coordinates)):
            self.save_plot(
                func=self.plot_partition_assigned_cameras,
                path=osp.join(fig_dir, "{}-partition.png".format(coordinates.get_str_id(partition_idx))),
                partition_idx=partition_idx,
                point_xyzs=point_cloud.points,
                point_rgbs=point_cloud.colors,
            )

    def get_extra_data(self):
        return {}

    def save_partitioning_results(self, output_path: str, image_set: ImageSet):
        partition_dir = osp.join(output_path, "partition_infos")
        os.makedirs(partition_dir, exist_ok=True)
        self.save(partition_dir, extra_data=self.get_extra_data())

        is_images_assigned_to_partitions = torch.logical_or(
            self.is_camera_in_partition, self.is_partitions_visible_to_cameras
        )
        written_idx_list = []
        for partition_idx in tqdm(
            list(range(is_images_assigned_to_partitions.shape[0])), desc="Saving image lists and cameras"
        ):
            partition_image_indices = is_images_assigned_to_partitions[partition_idx].nonzero().squeeze(-1).tolist()
            partition_id_str = self.partition_coordinates.get_str_id(partition_idx)
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

    def set_plot_ax_limit(self, ax, plot_enlarge: float = 0.25):
        enlarged_min = self.scene_bounding_box.bounding_box.min - plot_enlarge * (
            self.scene_bounding_box.bounding_box.max - self.scene_bounding_box.bounding_box.min
        )
        enlarged_max = self.scene_bounding_box.bounding_box.max + plot_enlarge * (
            self.scene_bounding_box.bounding_box.max - self.scene_bounding_box.bounding_box.min
        )
        ax.set_xlim([enlarged_min[0], enlarged_max[0]])
        ax.set_ylim([enlarged_min[1], enlarged_max[1]])

    def save_plot(self, func: Callable, path: str, *args, **kwargs):
        fig, ax = plt.subplots()
        func(ax, *args, **kwargs)
        plt.savefig(path, dpi=600)
        plt.close(fig)
