import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

from internal.cameras.cameras import Camera, Cameras, CameraType
from internal.dataparsers import (DataParser, DataParserConfig,
                                  DataParserOutputs)
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataset import Dataset
from myimpl.utils.dataset_utils import (ExtraDataProcessor,
                                        ExtraDataProcessorContainer)


@dataclass
class NeighborCamera:
    reference_id: int = -1

    neighbor_indices: List[int] = field(default_factory=lambda: [])


@dataclass
class NeighborCameraOutputs:
    camera: Camera

    image_name: str

    image: torch.Tensor

    mask: torch.Tensor

    def get_two_view_geometry_properties(self, cam_ref: Camera):
        K_ref = cam_ref.get_K()
        K_neighbor = self.camera.get_K()

        ref2neighbor = torch.linalg.inv(cam_ref.world_to_camera) @ self.camera.world_to_camera  # transposed
        return ref2neighbor, K_ref, K_neighbor


class NeighborCameraProcesser(ExtraDataProcessor):
    def __init__(self):
        super().__init__()
        self.dataset: Dataset = None

    def update_properties(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self, data: NeighborCamera) -> NeighborCameraOutputs:
        if len(data.neighbor_indices) <= 0:
            return None

        neighbor_id = random.choice(data.neighbor_indices)
        (
            neighbor_name,
            neighbor_image,
            neighbor_mask,
        ) = self.dataset.get_image(index=neighbor_id)

        return NeighborCameraOutputs(
            camera=self.dataset.image_cameras[neighbor_id],
            image_name=neighbor_name,
            image=neighbor_image,
            mask=neighbor_mask,
        )


@dataclass
class ViewGraph(DataParserConfig):
    max_neighbor_num: int = 8

    min_angle: float = math.pi / 6  # 30 degree

    min_rel_position: float = 0.1

    max_rel_position: float = 0.8

    def instantiate(self, path, output_path, global_rank):
        return ViewGraphDataparser(path, output_path, global_rank, self)


class ViewGraphDataparser(DataParser):
    def __init__(self, path, output_path, global_rank, params: ViewGraph):
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def get_outputs(self):
        dataparser_outputs: DataParserOutputs = super().get_outputs()

        rel_dist, rel_angle = ViewGraphUtils.calc_rel_pose(dataparser_outputs.train_set.cameras)
        n_cams = len(rel_dist)

        # TODO: rel_position thresh -> real thresh, maybe use camera_extent
        keep_mask = ~torch.eye(n_cams, dtype=torch.bool).to(rel_dist.device)
        keep_mask = torch.logical_and(
            torch.logical_and(keep_mask, rel_angle < self.params.min_angle),
            torch.logical_and(
                rel_dist > self.params.min_rel_position,
                rel_dist < self.params.max_rel_position,
            ),
        )

        for cam_id in range(n_cams):
            valid_indices = torch.where(keep_mask[cam_id])[0]
            neighbors = ViewGraphUtils.top_k_cameras(
                rel_dist[cam_id],
                rel_angle[cam_id],
                valid_indices,
                self.params.max_neighbor_num,
            )

            if dataparser_outputs.train_set.extra_data[cam_id] is None:
                dataparser_outputs.train_set.extra_data[cam_id] = {}
            dataparser_outputs.train_set.extra_data[cam_id].update(
                {"neighbor_cameras": NeighborCamera(cam_id, neighbors)}
            )

        if dataparser_outputs.train_set.extra_data_processor is None:
            dataparser_outputs.train_set.extra_data_processor = ExtraDataProcessorContainer() # fmt: skip
        dataparser_outputs.train_set.extra_data_processor.add_processor(
            {
                "neighbor_cameras": NeighborCameraProcesser(cameras=dataparser_outputs.train_set.cameras) # fmt: skip
            }
        )

        return dataparser_outputs


class ViewGraphUtils:
    @staticmethod
    def calc_rel_pose(cameras: Cameras):
        centers = cameras.camera_center
        rel_dist = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), dim=-1)

        optical_axes = cameras.R[:, 2, :]
        optical_axes = torch.clamp(optical_axes / optical_axes.norm(dim=-1, keepdim=True), -1.0, 1.0)
        axes_dot = torch.einsum("ik, jk -> ij", optical_axes, optical_axes)
        rel_angle = torch.acos(axes_dot)
        return rel_dist, rel_angle

    @staticmethod
    def top_k_cameras(
        rel_dist: torch.Tensor,
        rel_angle: torch.Tensor,
        valid_indices: torch.Tensor,
        max_neighbor_num: int,
    ):
        if len(valid_indices) == 0:
            return []
        else:
            valid_dist = rel_dist[valid_indices]
            valid_angle = rel_angle[valid_indices]
            dist_rank = torch.argsort(valid_dist)
            angle_rank = torch.argsort(valid_angle)
            rank_score = torch.argsort(dist_rank + angle_rank)

            top_k_indices = valid_indices[rank_score[:max_neighbor_num]]
            neighbors = top_k_indices.tolist()
        return neighbors


@dataclass
class ColmapViewGraph(Colmap, ViewGraph):
    def instantiate(self, path, output_path, global_rank):
        return ColmapViewGraphDataparser(path, output_path, global_rank, self)


class ColmapViewGraphDataparser(ColmapDataParser, ViewGraphDataparser):
    pass
