import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch


class ExtraData(ABC):
    KEY = ""
    """use for indexing extra data in ExtraDataContainer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExtraDataProcessor(ABC, Callable):
    KEY = ""
    """use for indexing extra data processor in ExtraDataContainer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, extra_data: Dict[str, Any], *args, **kwargs):
        pass

    def update_properties(self, *args, **kwargs):
        pass

    @classmethod
    def collate_fn(cls, extra_data_list: list):
        return torch.stack(extra_data_list, dim=0)


class ExtraDataContainer(dict):
    def add_extra_data(self, extra_data: ExtraData):
        self[extra_data.KEY] = extra_data


class ExtraDataProcessorContainer(dict):
    def add_extra_data_processor(self, extra_data_processor: ExtraDataProcessor):
        self[extra_data_processor.KEY] = extra_data_processor

    def update_properties(self, *args, **kwargs):
        for v in self.values():
            v.update_properties(*args, **kwargs)

    def __call__(self, extra_data: ExtraDataContainer) -> "ExtraDataProcessorOutputs":
        results = ExtraDataProcessorOutputs()
        for k, v in extra_data.items():
            if k in self:
                results.update({k: self[k](v)})
            else:
                results.update({k: v})
        return results


class ExtraDataProcessorOutputs(dict):
    pass


class NumpyData(ExtraData):
    def __init__(self, filepath: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filepath = filepath

    def parse_numpy_shape_from_header(self) -> List[int]:
        assert osp.exists(self.filepath) and self.filepath.endswith(".npy"), f"illegal filepath {self.filepath}"

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


class NumpyDataProcessor(ExtraDataProcessor):
    def __init__(self, device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def update_properties(self, *args, **kwargs):
        if "dataset" in kwargs:
            self.device = kwargs["dataset"].image_device

    def __call__(self, extra_data: NumpyData):
        try:
            data = torch.from_numpy(np.load(extra_data.filepath)).to(self.device)
        except:
            data = None
        return data


class SemanticData(NumpyData):
    KEY: str = "semantic"


class SemanticDataProcessor(NumpyDataProcessor):
    KEY: str = "semantic"

    def __init__(self, semantic_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_dim = semantic_dim

    def update_properties(self, *args, **kwargs):
        super().update_properties(*args, **kwargs)
        if "lambda_feature" in kwargs:
            self.lambda_feature = kwargs["lambda_feature"]

    def __call__(self, extra_data: SemanticData, *args, **kwargs):
        if getattr(self.lambda_feature, "enabled", False):
            return super().__call__(extra_data)
        return None


class DepthData(NumpyData):
    KEY: str = "depth"

    def __init__(self, filepath: str, img_shape: Tuple[int], scale: float = 1.0, offset: float = 0.0, *args, **kwargs):
        super().__init__(filepath, *args, **kwargs)
        self.img_shape = img_shape
        self.scale = scale
        self.offset = offset


class DepthDataProcessor(NumpyDataProcessor):
    KEY: str = "depth"

    def __call__(self, extra_data: DepthData, *args, **kwargs):
        depth: torch.Tensor = super().__call__(extra_data)
        if depth is None:
            return None
        depth = depth * extra_data.scale + extra_data.offset
        depth = torch.clamp_min(depth, min=0.0)
        if depth.shape != extra_data.img_shape:
            depth = torch.nn.functional.interpolate(
                depth[None, None, ...],
                size=extra_data.img_shape,
                mode="bilinear",
                align_corners=True,
            )[0][0]
        return depth


class MaskData(NumpyData):
    KEY: str = "mask"

    def __init__(self, filepath: str, img_shape: Tuple[int], *args, **kwargs):
        super().__init__(filepath, *args, **kwargs)
        self.img_shape = img_shape


class MaskDataProcessor(NumpyDataProcessor):
    KEY: str = "mask"

    def __call__(self, extra_data: MaskData):
        mask: torch.Tensor = super().__call__(extra_data)
        if mask is None:
            return None
        mask = mask.float()
        if mask.shape != extra_data.img_shape:
            mask = torch.nn.functional.interpolate(
                mask[None, None, ...],
                size=extra_data.img_shape,
                mode="bilinear",
                align_corners=True,
            )[0][0]
        return mask
