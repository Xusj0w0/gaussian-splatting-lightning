import os
import os.path as osp
import sys

from jsonargparse import ArgumentParser, set_docstring_parse_options

from large_scene.utils.train.training_utils import (
    ModifiedPartitionTraining, ModifiedPartitionTrainingConfig)

set_docstring_parse_options(attribute_docstrings=True)


class CityGSParitionTraining(ModifiedPartitionTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_partition_specific_args(self, partition_idx):
        return super().get_partition_specific_args(partition_idx) + [
            "--model.initialize_from={}".format(
                osp.join(self.path, "partitions", self.get_partition_id_str(partition_idx), "gaussian_model.ply")
            )
        ]


def main():
    parser = ArgumentParser()
    ModifiedPartitionTrainingConfig.configure_argparser(parser)
    CityGSParitionTraining.start_with_configured_argparser(parser, config_cls=ModifiedPartitionTrainingConfig)


if __name__ == "__main__":
    main()
