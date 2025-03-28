import argparse

from large_scene.utils.partition_training import (PartitionTraining,
                                                  PartitionTrainingConfig)


def main():
    parser = argparse.ArgumentParser()
    PartitionTrainingConfig.configure_argparser(parser)
    PartitionTraining.start_with_configured_argparser(parser, config_cls=PartitionTrainingConfig)


if __name__ == "__main__":
    main()
