import jsonargparse

from large_scene.partition_utils.utils.train.training_utils import (
    ModifiedPartitionTraining, ModifiedPartitionTrainingConfig)


def main():
    parser = jsonargparse.ArgumentParser()
    ModifiedPartitionTrainingConfig.configure_argparser(parser)
    ModifiedPartitionTraining.start_with_configured_argparser(parser, config_cls=ModifiedPartitionTrainingConfig)


if __name__ == "__main__":
    main()
