import argparse
import os
from dataclasses import dataclass

import torch
from external.partition_utils import VastGSPartitionCoordinates

from utils.train_partitions import PartitionTraining, PartitionTrainingConfig


@dataclass
class VastGSPartitionTrainingConfig(PartitionTrainingConfig):
    eval: bool = False
    dataset_path: str = ""

    @classmethod
    def get_extra_init_kwargs(cls, args):
        return {"eval": args.eval, "dataset_path": args.dataset_path}

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        PartitionTrainingConfig.configure_argparser(parser, extra_epoches)
        parser.add_argument("--eval", action="store_true", default=False)
        parser.add_argument("--dataset_path", default="")


class VastGSPartitionTraining(PartitionTraining):
    def __init__(self, config: PartitionTrainingConfig, name: str = "partitions.pt"):
        self.path = config.partition_dir
        self.config = config
        self.scene = torch.load(os.path.join(self.path, name), map_location="cpu")
        self.scene["partition_coordinates"] = VastGSPartitionCoordinates(**self.scene["partition_coordinates"])
        self.dataset_path = (
            config.dataset_path if len(config.dataset_path) > 0 else os.path.dirname(self.path.rstrip("/"))
        )

    def get_image_numbers(self) -> torch.Tensor:
        return self.scene["final_assignments"].sum(-1)

    def get_partition_image_number(self, partition_idx: int) -> int:
        return self.scene["final_assignments"][partition_idx].sum(-1).item()

    def get_default_dataparser_name(self) -> str:
        return "Colmap"

    def get_dataset_specific_args(self, partition_idx: int) -> list[str]:
        return [
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
            "--data.parser.points_from=ply",
            "--data.parser.ply_file={}".format(
                os.path.join(self.path, "init_pcds", f"{self.get_partition_id_str(partition_idx)}.ply")
            ),
        ]


def main():
    parser = argparse.ArgumentParser()
    VastGSPartitionTrainingConfig.configure_argparser(parser)
    VastGSPartitionTraining.start_with_configured_argparser(parser, config_cls=VastGSPartitionTrainingConfig)


if __name__ == "__main__":
    main()
