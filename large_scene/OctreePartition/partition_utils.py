import math
import os
import os.path as osp
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera, Cameras
from internal.utils.partitioning_utils import (
    MinMaxBoundingBox,
    MinMaxBoundingBoxes,
    PartitionableScene,
    PartitionCoordinates,
    Partitioning,
    SceneBoundingBox,
    SceneConfig,
)
from large_scene.VastGaussian.utils.partitioning_utils import VastGSScene, VastGSSceneConfig


@dataclass
class OctreeSceneConfig(VastGSSceneConfig):
    visibility_threshold: float = 0.0


@dataclass
class OctreeScene(VastGSScene):
    def calculate_camera_visibilities(self):
        pass
