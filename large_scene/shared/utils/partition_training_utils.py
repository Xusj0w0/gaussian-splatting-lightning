import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(os.getcwd(), "utils"))
from dataclasses import dataclass

import torch

from internal.utils.partitioning_utils import PartitionCoordinates
from utils.train_partitions import PartitionTraining, PartitionTrainingConfig


@dataclass
class ModifiedPartitionTrainingConfig(PartitionTrainingConfig):
    """
    1. Reorganize the directory structure of partition training.
    2. Dataset path should be manually specified.
    3. Image split:
        - `image_list` is set to the cameras in partition coordinates (including train and valid cameras);
        - `eval_image_select_mode` = \"list\" and `eval_list` = <path/to/valid_cameras.txt>.
        - `split_mode` = \"experiment\", ensuring images for training and validation are non-overlap.
    """

    eval: bool = False
    dataset_path: str = ""
    """ specify dataset path """

    @classmethod
    def get_extra_init_kwargs(cls, args):
        return {"eval": args.eval, "dataset_path": args.dataset_path}

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        PartitionTrainingConfig.configure_argparser(parser, extra_epoches)
        parser.add_argument("--eval", action="store_true", default=False)
        parser.add_argument("--dataset_path", type=str, default="")


class ModifiedPartitionTraining(PartitionTraining):
    def __init__(self, config: PartitionTrainingConfig, name: str = "partitions.pt"):
        self.path = config.partition_dir
        self.config = config
        self.scene = torch.load(os.path.join(self.path, name), map_location="cpu")
        self.scene["partition_coordinates"] = PartitionCoordinates(**self.scene["partition_coordinates"])
        self.dataset_path = (
            config.dataset_path if len(config.dataset_path) > 0 else os.path.dirname(self.path.rstrip("/"))
        )

    def get_default_dataparser_name(self) -> str:
        return "Colmap"

    def get_dataset_specific_args(self, partition_idx: int) -> list[str]:
        return super().get_dataset_specific_args(partition_idx) + [
            "--data.parser.image_list={}".format(
                os.path.join(
                    self.path,
                    "image_lists",
                    "{}.txt".format(self.get_partition_id_str(partition_idx)),
                )
            ),
            "--data.parser.eval_image_select_mode=list",
            "--data.parser.eval_list={}".format(os.path.join(self.dataset_path, "splits/val_images.txt")),
            "--data.parser.split_mode={}".format("experiment" if self.config.eval else "reconstruction"),
        ]

    @property
    def project_output_dir(self) -> str:
        """Append \'partitions\' after project name: <workdir>/outputs/<project>/partitions"""
        return osp.join(super().project_output_dir, "partitions")
