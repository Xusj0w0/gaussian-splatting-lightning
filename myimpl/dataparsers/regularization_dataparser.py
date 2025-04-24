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
from myimpl.dataparsers.extra_data_processors import (
    EstimatedInvDepthConfigMixin, EstimatedInvDepthMixin, MaskConfigMixin,
    MaskMixin, SemanticConfigMixin, SemanticMixin)
from myimpl.utils.dataset_utils import (ExtraDataContainer,
                                        ExtraDataProcessorContainer)


@dataclass
class Regularization(Colmap, EstimatedInvDepthConfigMixin, SemanticConfigMixin):
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

        return dataparser_outputs
