import argparse
from utils.partition_training_utils import ModifiedPartitionTraining, ModifiedPartitionTrainingConfig


def main():
    parser = argparse.ArgumentParser()
    ModifiedPartitionTrainingConfig.configure_argparser(parser)
    ModifiedPartitionTraining.start_with_configured_argparser(parser, config_cls=ModifiedPartitionTrainingConfig)


if __name__ == "__main__":
    main()
