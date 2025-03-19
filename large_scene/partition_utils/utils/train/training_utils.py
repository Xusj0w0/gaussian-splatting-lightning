import os
import os.path as osp
from dataclasses import dataclass, field

import yaml
from jsonargparse import ArgumentParser

from utils.argparser_utils import parser_stoppable_args, split_stoppable_args
from utils.distibuted_tasks import configure_arg_parser_v2

from .base import PartitionTraining, PartitionTrainingConfig


# fmt: off
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
    partition_dir: str = field(init=False)
    eval: bool = True
    no_default_scalable: bool = False
    scalable_config: str = field(
        default_factory=lambda: osp.normpath(
            osp.join(__file__, "../..", "scalable_param_configs/default.yaml")
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.partition_dir = osp.normpath(
            osp.join(__file__, "../../../../..", "outputs", self.project_name, "partition_infos")
        )

    @classmethod
    def configure_argparser(cls, parser: ArgumentParser, extra_epoches: int = 0):
        parser.add_class_arguments(cls)

        # modify parser
        container = ArgumentParser()
        # _parser.add_argument("partition_dir")
        container.add_argument("--project-name", "-p", type=str, required=True,
                            help="Project name")
        container.add_argument("--eval", action="store_true")
        container.add_argument("--min-images", "-m", type=int, default=32,
                            help="Ignore partitions with image number less than this value")
        container.add_argument("--config-file", "-c", type=str, nargs="*", default=None)
        container.add_argument("--partition_id_strs", default=None, nargs="*", action="extend")
        container.add_argument("--extra-epoches", "-e", type=int, default=extra_epoches)
        container.add_argument("--scalable-config", type=str, default=None,
                            help="Load scalable params from a yaml file")
        container.add_argument("--scale-base", type=int, default=300)
        container.add_argument("--scalable-params", default=[], nargs="*", action="extend")
        container.add_argument("--extra-epoch-scalable-params", default=[], nargs="*", action="extend")
        container.add_argument("--scale-param-mode", type=str, default="linear")
        container.add_argument("--max-steps", type=int, default=30_000)
        container.add_argument("--no-default-scalable", action="store_true")
        container.add_argument("--dry-run", action="store_true", default=False)
        container.add_argument("--name-suffix", default="")
        container.add_argument("--ff-densify", action="store_true", default=False)
        container.add_argument("--t3dgs-densify", action="store_true", default=False)
        container.add_argument("--image-number-from", type=str, default=None)
        configure_arg_parser_v2(container)

        def return_default(a, b):
            return b if a is None else a

        action_dict = {action.dest: action for action in container._actions if "help" not in action.dest}
        for i, action in enumerate(parser._actions):
            if action.dest in action_dict:
                for property in ["option_strings", "_typehint", "required", "nargs", "choices", "const"]:
                    setattr(parser._actions[i], property, getattr(action_dict[action.dest], property, None))
                parser._actions[i].help = return_default(action_dict[action.dest].help, action.help)
                parser._actions[i].default = return_default(action_dict[action.dest].default, action.default)

    @classmethod
    def instantiate_with_args(cls, parser: ArgumentParser, args, training_args, srun_args):
        scale_base, max_steps, scalable_params, extra_epoch_scalable_params, scale_param_mode = cls.parse_scalable_params(args)

        cfg = parser.instantiate_classes(args)
        config = cls(**vars(cfg))
        config.max_steps = max_steps
        config.scale_base = scale_base
        config.scalable_params = scalable_params
        config.extra_epoch_scalable_params = extra_epoch_scalable_params
        config.scale_param_mode = scale_param_mode
        config.training_args = training_args
        config.srun_args = srun_args

        return config

    @classmethod
    def instantiate_with_parser(cls, parser):
        args, training_and_srun_args = parser_stoppable_args(parser)
        training_args, srun_args = split_stoppable_args(training_and_srun_args)
        return cls.instantiate_with_args(parser, args, training_args, srun_args), args


class ModifiedPartitionTraining(PartitionTraining):
    config: ModifiedPartitionTrainingConfig

    def __init__(self, config: PartitionTrainingConfig, name: str = "partitions.pt"):
        super().__init__(config, name)

        part_config_path = osp.join(config.partition_dir, "config.yaml")
        self.dataset_path = yaml.safe_load(open(part_config_path, "r"))["dataset_path"]

    def get_default_dataparser_name(self) -> str:
        return "Colmap"

    def get_dataset_specific_args(self, partition_idx: int) -> list[str]:
        return super().get_dataset_specific_args(partition_idx) + [
            "--data.parser.image_list={}".format(
                os.path.join(self.path, "partitions", self.get_partition_id_str(partition_idx), "image_list.txt")
            ),
            "--data.parser.eval_image_select_mode=list",
            "--data.parser.eval_list={}".format(os.path.join(self.dataset_path, "splits/val_images.txt")),
            "--data.parser.split_mode={}".format("experiment" if self.config.eval else "reconstruction"),
        ]

    @property
    def project_output_dir(self) -> str:
        """Append \'partitions\' after project name: <workdir>/outputs/<project>/partitions"""
        return osp.join(super().project_output_dir, "partitions")

    @staticmethod
    def get_project_output_dir_by_name(project_name: str) -> str:
        return osp.join(
            osp.normpath(osp.join(osp.dirname(__file__), "../../../..")),
            "outputs", project_name,
        )
