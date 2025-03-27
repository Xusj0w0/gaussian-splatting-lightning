from jsonargparse import ArgumentParser, set_docstring_parse_options

from large_scene.utils.partition_training import (PartitionTraining,
                                                  PartitionTrainingConfig)

set_docstring_parse_options(attribute_docstrings=True)


def main():
    parser = ArgumentParser()
    PartitionTrainingConfig.configure_argparser(parser)
    PartitionTraining.start_with_configured_argparser(parser, config_cls=PartitionTrainingConfig)


if __name__ == "__main__":
    main()
