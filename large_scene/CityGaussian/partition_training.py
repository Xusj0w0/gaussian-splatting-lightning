import argparse
import os
import os.path as osp
import sys
from dataclasses import dataclass

sys.path.insert(0, osp.join(os.getcwd(), "utils"))
import torch

from internal.utils.partitioning_utils import PartitionCoordinates

from ..shared.utils.partition_training_utils import (
    ModifiedPartitionTraining, ModifiedPartitionTrainingConfig)


class CityGSParitionTraining(ModifiedPartitionTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_partition_specific_args(self, partition_idx):
        return super().get_partition_specific_args(partition_idx) + [
            "--model.initialize_from={}".format(
                osp.join(self.path, "partition_infos", self.get_partition_id_str(partition_idx), "gaussian_model.ply")
            )
        ]
