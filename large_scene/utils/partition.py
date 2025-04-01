import glob
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from jsonargparse import ArgumentParser, Namespace, lazy_instance

from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.partitioning_utils import MinMaxBoundingBox
from large_scene.impls.base import PartitionableScene, PartitionableSceneConfig
from large_scene.impls.city_gaussian import (CityScene, CitySceneConfig,
                                             UncontractedCityScene,
                                             UncontractedCitySceneConfig)
from large_scene.impls.vast_gaussian import VastScene, VastSceneConfig

__all__ = ["Partition"]


class Partition:
    def __init__(
        self,
        project_name: str,
        dataset_path: str,
        scene_config: PartitionableSceneConfig = lazy_instance(PartitionableSceneConfig),
        *args,
        **kwargs,
    ):
        self.project_name = project_name
        self.dataset_path = dataset_path
        self.scene_config = scene_config

        self.output_path = osp.join("outputs", self.project_name)
        os.makedirs(osp.join(self.output_path, "partition_infos"), exist_ok=True)

        self.scene = self.scene_config.instantiate()

    def partition(self):
        self.scene.partition(self.dataset_path, self.output_path)

    @classmethod
    def start(cls, parser: ArgumentParser):
        cls.configure_argparser(parser)
        args = parser.parse_args()
        args = cls.parse_manhattan(args)
        cfg = parser.instantiate_classes(args)
        partitioning = cls(**cfg)

        # save yaml config
        parser.save(args, osp.join(partitioning.output_path, "partition_infos/config.yaml"), overwrite=True)

        partitioning.partition()

    @classmethod
    def configure_argparser(cls, parser: ArgumentParser):
        parser.add_class_arguments(cls, nested_key=None)
        parser.add_argument("--manhattan_trans", type=Optional[Union[str, List[float]]])
        parser.add_argument("--partition_dim", type=List[int], required=True)
        parser.link_arguments("manhattan_trans", "scene_config.init_args.manhattan_trans", apply_on="parse")
        parser.link_arguments("partition_dim", "scene_config.init_args.partition_dim", apply_on="parse")
        return parser

    @classmethod
    def parse_manhattan(cls, args):
        default = [1.0 if i == j else 0.0 for j in range(4) for i in range(4)]
        manhattan_trans = args.manhattan_trans
        if manhattan_trans is None:
            manhattan_path = glob.glob(osp.join(args.dataset_path, "**", "manhattan.txt"), recursive=True)
            if len(manhattan_path) > 0:
                manhattan_path = manhattan_path[0]
            try:
                with open(manhattan_path, "r") as fid:
                    manhattan_trans = []
                    for line in fid.readlines():
                        manhattan_trans += map(float, line.strip().split())
            except:
                manhattan_trans = default
        else:
            manhattan_trans = default

        try:
            manhattan_np = np.array(manhattan_trans).reshape(4, 4)
            rot = manhattan_np[:3, :3]
            vec = np.zeros((4,), dtype=rot.dtype)
            vec[-1] = 1.0
            assert np.allclose(rot @ rot.T, np.eye(3, dtype=rot.dtype), atol=1e-6) and np.isclose(
                np.linalg.det(rot), 1.0, atol=1e-6
            ), "3x3 sub matrix should be Rotation Matrix"
            assert np.allclose(manhattan_np[-1, :], vec, atol=1e-6), "last row should be [0, 0, 0, 1]"
        except:
            manhattan_trans = default

        args.manhattan_trans = manhattan_trans
        args.scene_config.init_args.manhattan_trans = manhattan_trans
        return args

    @classmethod
    def load_partition(cls, config_path: str):
        parser = ArgumentParser()
        cls.configure_argparser(parser)
        args = parser.parse_path(config_path)
        cfg = parser.instantiate_classes(args)
        return cls(**cfg)
