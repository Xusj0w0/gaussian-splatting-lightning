import json
import math
import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers import DataParser, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from myimpl.dataparsers.extra_data_processors import (DepthData,
                                                      DepthDataProcessor,
                                                      SemanticData,
                                                      SemanticDataProcessor)
from myimpl.utils.dataset_utils import (ExtraData, ExtraDataContainer,
                                        ExtraDataProcessor,
                                        ExtraDataProcessorContainer)


@dataclass
class EstimatedInvDepthConfigMixin:
    depth_dir: str = "extra/estimated_invdepth"

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
            with open(osp.join(self.path, self.params.depth_scale_filename), "r") as f:
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
                depth_file_path = osp.join(self.path, self.params.depth_dir, f"{image_name}.npy")
                if osp.exists(depth_file_path) is False:
                    print("[WARNING] {} does not have a depth file".format(image_name))
                    continue

                depth_scale = {"scale": 1.0, "offset": 0.0}
                if self.params.depth_rescaling:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is None:
                        print("[WARNING {} does not have a depth scale]".format(image_name))
                        continue
                    if (
                        depth_scale["scale"] < self.params.depth_scale_lower_bound * median_scale
                        or depth_scale["scale"] > self.params.depth_scale_upper_bound * median_scale
                    ):
                        print(
                            "[WARNING depth scale '{}' of '{}' out of bound '({}, {})']".format(
                                depth_scale["scale"],
                                image_name,
                                self.params.depth_scale_lower_bound * median_scale,
                                self.params.depth_scale_upper_bound * median_scale,
                            )
                        )
                        continue

                if not isinstance(image_set.extra_data[idx], ExtraDataContainer):
                    image_set.extra_data[idx] = ExtraDataContainer()
                image_set.extra_data[idx].add_extra_data(
                    DepthData(depth_file_path, depth_scale["scale"], depth_scale["offset"])
                )

            if not isinstance(image_set.extra_data_processor, ExtraDataProcessorContainer):
                image_set.extra_data_processor = ExtraDataProcessorContainer()
            image_set.extra_data_processor.add_extra_data_processor(DepthDataProcessor())


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
                if osp.exists(semantic_file_path) is False:
                    print("[WARNING] {} does not have a semantic file".format(image_name))
                    continue

                if not isinstance(image_set.extra_data[idx], ExtraDataContainer):
                    image_set.extra_data[idx] = ExtraDataContainer()
                image_set.extra_data[idx].add_extra_data(SemanticData(semantic_file_path))

            if not isinstance(image_set.extra_data_processor, ExtraDataProcessorContainer):
                image_set.extra_data_processor = ExtraDataProcessorContainer()
            image_set.extra_data_processor.add_extra_data_processor(SemanticDataProcessor())


@dataclass
class RegularizationDataParserConfig(Colmap, EstimatedInvDepthConfigMixin, SemanticConfigMixin):
    def instantiate(self, path, output_path, global_rank):
        return RegularizationDataParser(path, output_path, global_rank, self)


class RegularizationDataParser(ColmapDataParser, EstimatedInvDepthMixin, SemanticMixin):
    def __init__(self, path: str, output_path: str, global_rank: int, params: RegularizationDataParserConfig) -> None:
        self.params: RegularizationDataParserConfig
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()
        self.configure_depth_data(dataparser_outputs)
        self.configure_semantic_data(dataparser_outputs)
        return dataparser_outputs
