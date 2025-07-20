import json
import math
import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers import DataParser, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from myimpl.dataparsers.extra_dataparsers import (EstimatedInvDepthConfigMixin,
                                                  EstimatedInvDepthMixin,
                                                  MaskConfigMixin, MaskMixin,
                                                  SemanticConfigMixin,
                                                  SemanticMixin)
from myimpl.utils.dataset_utils import (ExtraDataContainer,
                                        ExtraDataProcessorContainer)


@dataclass
class Regularization(Colmap, EstimatedInvDepthConfigMixin, SemanticConfigMixin):
    camera_extent_all_images: bool = False

    def instantiate(self, path, output_path, global_rank):
        return RegularizationDataParser(path, output_path, global_rank, self)


class RegularizationDataParser(ColmapDataParser, EstimatedInvDepthMixin, SemanticMixin):
    def __init__(self, path: str, output_path: str, global_rank: int, params: Regularization) -> None:
        self.params: Regularization
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            if not isinstance(image_set.extra_data_processor, ExtraDataProcessorContainer):
                image_set.extra_data_processor = ExtraDataProcessorContainer()
            for idx in range(len(image_set.image_names)):
                if not isinstance(image_set.extra_data[idx], ExtraDataContainer):
                    image_set.extra_data[idx] = ExtraDataContainer()

        self.configure_depth_data(dataparser_outputs)
        self.configure_semantic_data(dataparser_outputs)

        # get camera extent from all images
        if self.params.camera_extent_all_images:
            colmap_params = Colmap(**{k: getattr(self.params, k) for k in Colmap.__dataclass_fields__})
            colmap_params.image_list = None
            colmap_params.points_from = "random"
            colmap_params.n_random_points = 1
            colmap = colmap_params.instantiate(
                path=self.path, output_path=self.output_path, global_rank=self.global_rank
            )
            outputs = colmap.get_outputs()
            dataparser_outputs.camera_extent = outputs.camera_extent

            # images = colmap_utils.read_images_binary(os.path.join(self.detect_sparse_model_dir(), "images.bin"))
            # Rs, Ts = [], []
            # for idx, key in enumerate(images):
            #     extrinsics = images[key]
            #     Rs.append(extrinsics.qvec2rotmat())
            #     Ts.append(np.array(extrinsics.tvec))
            # R = torch.tensor(np.stack(Rs, axis=0), dtype=torch.float32)
            # T = torch.tensor(np.stack(Ts, axis=0), dtype=torch.float32)
            # w2c = torch.zeros(size=(R.shape[0], 4, 4))
            # w2c[:, :3, :3] = R
            # w2c[:, :3, 3] = T
            # w2c[:, 3, 3] = 1.0
            # c2w = torch.linalg.inv(w2c)
            # camera_centers = c2w[:, :3, 3]
            # average_camera_center = torch.mean(camera_centers, dim=0)
            # camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
            # max_distance = torch.max(camera_distance)
            # dataparser_outputs.camera_extent = float(max_distance * 1.1)

        return dataparser_outputs
