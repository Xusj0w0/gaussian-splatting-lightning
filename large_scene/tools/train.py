from jsonargparse import ArgumentParser, set_docstring_parse_options

from large_scene.utils.train.training_utils import (
    ModifiedPartitionTraining, ModifiedPartitionTrainingConfig)

set_docstring_parse_options(attribute_docstrings=True)


def main():
    parser = ArgumentParser()
    ModifiedPartitionTrainingConfig.configure_argparser(parser)
    ModifiedPartitionTraining.start_with_configured_argparser(parser, config_cls=ModifiedPartitionTrainingConfig)


if __name__ == "__main__":
    main()
