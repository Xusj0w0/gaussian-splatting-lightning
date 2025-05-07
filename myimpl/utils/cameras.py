from dataclasses import dataclass, field

import torch
from torch import Tensor

from internal.cameras.cameras import Camera, Cameras, CameraType


@dataclass
class InstantiatedCameras(Cameras):
    fov_x: Tensor = field(init=True)
    fov_y: Tensor = field(init=True)
    world_to_camera: Tensor = field(init=True)
    projection: Tensor = field(init=True)
    full_projection: Tensor = field(init=True)
    camera_center: Tensor = field(init=True)

    def __post_init__(self):
        pass

    @property
    def device(self):
        return self.R.device

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            params = {}
            for k in self.__dataclass_fields__:
                v = getattr(self, k)
                try:
                    params[k] = v[key]
                except:
                    params[k] = None
            return InstantiatedCameras(**params)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
