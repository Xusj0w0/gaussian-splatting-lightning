import math
import os
import os.path as osp
from dataclasses import MISSING, asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers import DataParser, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from myimpl.utils.dataset_utils import (ExtraDataContainer, ExtraDataProcessor,
                                        ExtraDataProcessorContainer)


@dataclass
class FeatureShapeCamera(Camera):
    feature_shape: torch.Tensor = None

    def preprocess_feature_camera(self, render_feature_size: int):
        if self.width > self.height:
            h = int(render_feature_size)
            w = int(round(render_feature_size * float(self.width / self.height)))
        else:
            w = int(render_feature_size)
            h = int(round(render_feature_size * self.height / self.width))
        scale_x, scale_y = float(w) / self.width.item(), float(h) / self.height.item()

        viewmats = self.world_to_camera.T.unsqueeze(0)
        # fmt: off
        Ks = torch.tensor([[
            [self.fx * scale_x, 0, self.cx * scale_x],
            [0.0, self.fy * scale_y, self.cy * scale_y],
            [0.0, 0.0, 1.0]
        ]], dtype=torch.float, device=self.R.device)
        # fmt: on

        return viewmats, Ks, (w, h)


@dataclass
class FeatureShapeCameras(Cameras):
    feature_shape: torch.Tensor = None

    def __getitem__(self, index):
        camera = super().__getitem__(index)
        return FeatureShapeCamera(
            feature_shape=self.feature_shape[index], **{k: getattr(camera, k) for k in camera.__dataclass_fields__}
        )


@dataclass
class ExtraSemanticFeatureData:
    """
    Only support npy file, and shape should be HWC.
    """

    filepath: str
    shape: List[int] = field(init=False)

    def __post_init__(self):
        self.shape = self.parse_numpy_shape_from_header()

    def parse_numpy_shape_from_header(self) -> List[int]:
        with open(self.filepath, "rb") as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError("Not a valid .npy file")

            major, minor = np.frombuffer(f.read(2), dtype=np.uint8)
            if major == 1:
                header_len = np.frombuffer(f.read(2), dtype=np.uint16)[0]
            elif major == 2:
                header_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            else:
                raise ValueError("Unsupported .npy version")

            header = f.read(header_len).decode("latin1")
            header_dict = eval(header)

        return header_dict["shape"]

    def load_data(self) -> np.ndarray:
        if self.filepath.endswith(".npy"):
            data = np.load(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {self.filepath}")
        return data


class SemanticFeatureProcessor(ExtraDataProcessor):
    KEY: str = "semantic_feature"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cpu")

    def update_properties(self, dataset):
        self.device = dataset.image_device

    def __call__(self, data: ExtraSemanticFeatureData):
        torch_data = torch.from_numpy(data.load_data()).float()
        return torch_data.to(self.device)


@dataclass
class SemanticFeature(Colmap):
    feature_dir: str = "semantic"
    """feature path relative to dataparser.path, for example semantic/sam2_features"""

    filename_suffix: str = ".npy"

    filename_include_image_ext: bool = True

    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return FeatureDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class FeatureDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: SemanticFeature) -> None:
        self.params: SemanticFeature
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        key = "semantic_feature"

        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            n_cam = len(image_set.cameras)
            image_names = image_set.image_names
            feature_shapes = []

            for cam_idx in range(n_cam):
                if not isinstance(image_set.extra_data[cam_idx], ExtraDataContainer):
                    image_set.extra_data[cam_idx] = ExtraDataContainer()
                image_name = (
                    image_names[cam_idx]
                    if self.params.filename_include_image_ext
                    else image_names[cam_idx].rsplit(".", 1)[0]
                )
                filepath = osp.join(self.path, self.params.feature_dir, f"{image_name}{self.params.filename_suffix}")
                if not osp.exists(filepath):
                    raise FileNotFoundError(f"{filepath} not found")
                semantic_feature = ExtraSemanticFeatureData(filepath)

                image_set.extra_data[cam_idx].update({key: semantic_feature})
                feature_shapes.append(torch.tensor(semantic_feature.shape[:2]))

            # update cameras
            feature_shapes = torch.stack(feature_shapes, 0)
            image_set.cameras = FeatureShapeCameras(
                feature_shape=feature_shapes,
                **{
                    k: getattr(image_set.cameras, k)
                    for k, v in image_set.cameras.__dataclass_fields__.items()
                    if v.init
                },
            )

            if not isinstance(image_set.extra_data_processor, ExtraDataProcessorContainer):
                image_set.extra_data_processor = ExtraDataProcessorContainer()
            image_set.extra_data_processor.add_processor(key, SemanticFeatureProcessor())

        return dataparser_outputs
