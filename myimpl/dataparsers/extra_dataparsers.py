import json
import math
import os
import os.path as osp
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers import DataParser, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from myimpl.utils.dataset_utils import (DepthData, DepthDataProcessor,
                                        MaskData, MaskDataProcessor,
                                        SemanticData, SemanticDataProcessor)


@dataclass
class SemanticConfigMixin:
    semantic_dir: str = ""


class SemanticMixin:
    params: Union[Colmap, SemanticConfigMixin]

    def configure_semantic_data(
        self: Union["SemanticMixin", ColmapDataParser],
        dataparser_outputs: DataParserOutputs,
    ):
        if self.params.semantic_dir is None or len(self.params.semantic_dir) <= 0:
            return

        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                semantic_file_path = osp.join(self.path, self.params.semantic_dir, f"{image_name}.npy")
                if osp.exists(semantic_file_path):
                    extra_data = SemanticData(semantic_file_path)
                else:
                    extra_data = SemanticData(None)
                image_set.extra_data[idx].add_extra_data(extra_data)
            image_set.extra_data_processor.add_extra_data_processor(SemanticDataProcessor())


@dataclass
class EstimatedInvDepthConfigMixin:
    depth_dir: str = ""

    depth_rescaling: bool = True

    depth_scale_filename: str = "estimated_depth_scales.json"

    depth_scale_lower_bound: float = 0.2

    depth_scale_upper_bound: float = 5.0


class EstimatedInvDepthMixin:
    params: Union[Colmap, EstimatedInvDepthConfigMixin]

    def configure_depth_data(
        self: Union["EstimatedInvDepthMixin", ColmapDataParser],
        dataparser_outputs: DataParserOutputs,
    ):
        if self.params.depth_dir is None or len(self.params.depth_dir) <= 0:
            return

        if self.params.depth_rescaling:
            with open(osp.join(self.path, "extra", self.params.depth_scale_filename), "r") as f:
                depth_scales = json.load(f)
            image_name_set = {
                image_name: True
                for image_name in dataparser_outputs.train_set.image_names + dataparser_outputs.val_set.image_names
            }
            depth_scale_list = []
            for image_name, image_depth_scale in depth_scales.items():
                if image_name not in image_name_set:
                    continue
                depth_scale_list.append(image_depth_scale["scale"])

            median_scale = np.median(np.asarray(depth_scale_list))

        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                depth_file_path = osp.join(self.path, "extra", self.params.depth_dir, f"{image_name}.npy")
                # if not exist, set `depth_file_path` to None
                if osp.exists(depth_file_path) is False:
                    depth_file_path = None

                depth_scale = {"scale": 1.0, "offset": 0.0}
                # if not satisfy requirements, set `depth_scale` to None
                if depth_file_path is not None and self.params.depth_rescaling:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is not None and (
                        depth_scale["scale"] < self.params.depth_scale_lower_bound * median_scale
                        or depth_scale["scale"] > self.params.depth_scale_upper_bound * median_scale
                    ):
                        depth_scale = None

                if depth_file_path is not None and depth_scale is not None:
                    extra_data = DepthData(
                        depth_file_path,
                        (image_set.cameras[idx].height.item(), image_set.cameras[idx].width.item()),
                        depth_scale["scale"],
                        depth_scale["offset"],
                    )
                else:
                    extra_data = DepthData(None, None)
                image_set.extra_data[idx].add_extra_data(extra_data)
            image_set.extra_data_processor.add_extra_data_processor(DepthDataProcessor())


@dataclass
class MaskConfigMixin:
    mask_dir: str = ""


class MaskMixin:
    params: Union[Colmap, MaskConfigMixin]

    def configure_mask_data(
        self: Union["MaskMixin", ColmapDataParser],
        dataparser_outputs: DataParserOutputs,
    ):
        if self.params.mask_dir is None or len(self.params.mask_dir) <= 0:
            return

        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                mask_file_path = osp.join(self.path, self.params.mask_dir, f"{image_name}.npy")
                if osp.exists(mask_file_path):
                    extra_data = MaskData(
                        mask_file_path, (image_set.cameras[idx].height.item(), image_set.cameras[idx].width.item())
                    )
                else:
                    extra_data = MaskData(None, None)
                image_set.extra_data[idx].add_extra_data(extra_data)
            image_set.extra_data_processor.add_extra_data_processor(MaskDataProcessor())
