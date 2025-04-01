import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import yaml
from jsonargparse import ArgumentParser
from tqdm import tqdm

from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.partitioning_utils import MinMaxBoundingBox
from large_scene.impls.base.partition_training import \
    PartitionTraining as PartitionTrainingBase
from large_scene.impls.base.partition_training import \
    PartitionTrainingConfig as PartitionTrainingConfigBase
from utils.distibuted_tasks import configure_arg_parser_v2

from .partition import Partition

__all__ = ["PartitionTrainingConfig", "PartitionTraining"]

# fmt: off
@dataclass
class PartitionTrainingConfig(PartitionTrainingConfigBase):
    partition_dir: str = field(init=False)
    def __post_init__(self):
        super().__post_init__()
        self.partition_dir = osp.normpath(
            osp.join(__file__, "../../..", "outputs", self.project_name, "partition_infos")
        )

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        # TODO: replace with jsonargparse
        # parser.add_argument("partition_dir")
        parser.add_argument("--project", "-p", type=str, required=True,
                            help="Project name")
        parser.add_argument("--min-images", "-m", type=int, default=32,
                            help="Ignore partitions with image number less than this value")
        parser.add_argument("--config", "-c", type=str, nargs="*", default=None)
        parser.add_argument("--parts", default=None, nargs="*", action="extend")
        parser.add_argument("--extra-epoches", "-e", type=int, default=extra_epoches)
        parser.add_argument("--scalable-config", type=str, default=None,
                            help="Load scalable params from a yaml file")
        parser.add_argument("--scale-base", type=int, default=300)
        parser.add_argument("--scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--extra-epoch-scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--scale-param-mode", type=str, default="none")
        parser.add_argument("--max-steps", type=int, default=30_000)
        parser.add_argument("--no-default-scalable", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--name-suffix", type=str, default="")
        parser.add_argument("--ff-densify", action="store_true", default=False)
        parser.add_argument("--t3dgs-densify", action="store_true", default=False)
        parser.add_argument("--image-number-from", type=str, default=None)
        configure_arg_parser_v2(parser)

    @classmethod
    def instantiate_with_args(cls, args, training_args, srun_args):
        scale_base, max_steps, scalable_params, extra_epoch_scalable_params, scale_param_mode = cls.parse_scalable_params(args)

        return cls(
            # partition_dir=args.partition_dir,
            project_name=args.project,
            min_images=args.min_images,
            n_processes=args.n_processes,
            process_id=args.process_id,
            dry_run=args.dry_run,
            extra_epoches=args.extra_epoches,
            name_suffix=args.name_suffix,
            ff_densify=args.ff_densify,
            t3dgs_densify=args.t3dgs_densify,
            max_steps=max_steps,
            scale_base=scale_base,
            scalable_params=scalable_params,
            extra_epoch_scalable_params=extra_epoch_scalable_params,
            scale_param_mode=scale_param_mode,
            partition_id_strs=args.parts,
            config_file=args.config,
            image_number_from=args.image_number_from,
            training_args=training_args,
            srun_args=srun_args,
            **cls.get_extra_init_kwargs(args),
        )


class PartitionTraining(PartitionTrainingBase):
    config: PartitionTrainingConfig

    def __init__(self, config: PartitionTrainingConfigBase, name: str = "partitions.pt"):
        super().__init__(config, name)

        part_config_path = osp.join(config.partition_dir, "config.yaml")
        self.dataset_path = yaml.safe_load(open(part_config_path, "r"))["dataset_path"]

    def get_default_dataparser_name(self) -> str:
        return "Colmap"
    
    def get_partition_specific_args(self, partition_idx):
        args = super().get_partition_specific_args(partition_idx)
        args += [
            "--data.parser.image_list={}".format(
                osp.join(self.path, "partitions", self.get_partition_id_str(partition_idx), "image_list.txt")
            )
        ]

        # city gs init
        init_model_path = osp.join(self.path, "partitions", self.get_partition_id_str(partition_idx), "gaussian_model.ply")
        if osp.exists(init_model_path):
            args += [
                "--model.initialize_from={}".format(init_model_path),
            ]
        return args
 

    @property
    def project_output_dir(self) -> str:
        """Append \'partitions\' after project name: <workdir>/outputs/<project>/partitions"""
        return osp.join(super().project_output_dir, "partitions")

    @staticmethod
    def get_project_output_dir_by_name(project_name: str) -> str:
        return osp.join(
            osp.normpath(osp.join(__file__, "../../..")),
            "outputs", project_name,
        )

    def get_trained_partitions(self):
        trainable_partition_idx_list = self.get_trainable_partition_idx_list(
            min_images=self.config.min_images,
            n_processes=self.config.n_processes,
            process_id=self.config.process_id,
        )
        partition_bounding_boxes = self.partition_coordinates.get_bounding_boxes()

        trained_partitions = []
        for partition_idx in tqdm(trainable_partition_idx_list, desc="Searching checkpoints"):
            partition_id_str = self.get_partition_id_str(partition_idx)
            assert os.path.exists(
                os.path.join(
                    self.project_output_dir,
                    self.get_partition_trained_step_filename(partition_idx),
                )
            ), "partition {} not trained".format(partition_id_str)
            model_dir = os.path.join(self.project_output_dir, partition_id_str)
            ckpt_file = GaussianModelLoader.search_load_file(model_dir)
            assert ckpt_file.endswith(".ckpt"), "checkpoint of partition #{} ({}) can not be found in {}".format(
                partition_idx, partition_id_str, self.project_output_dir
            )
            trained_partitions.append(
                (
                    partition_idx,
                    partition_id_str,
                    ckpt_file,
                    partition_bounding_boxes[partition_idx],
                )
            )

        orientation_transformation = (
            self.scene["extra_data"]["rotation_transform"]
            if self.scene["extra_data"] is not None
            else None
        )
        return trained_partitions, orientation_transformation

    def split_partition_gaussians(
        self, ckpt: dict, partition_bounding_box: MinMaxBoundingBox
    ) -> tuple[
        VanillaGaussianModel,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        if getattr(self, "partition", None) is None:
            self.partition = Partition.load_partition(osp.join(self.config.partition_dir, "config.yaml"))
        
        model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, "cpu")
        kwargs = {'manhattan_trans': self.scene["extra_data"]["rotation_transform"]}
        if "radius_bounding_box" in self.scene["extra_data"]:
            kwargs.update({"radius_bbox": MinMaxBoundingBox(**self.scene["extra_data"]["radius_bounding_box"])})
        is_in_partition = self.partition.scene.is_in_partition(model.get_means(), partition_bounding_box, **kwargs)

        inside_part = {}
        outside_part = {}
        for k, v in model.properties.items():
            inside_part[k] = v[is_in_partition]
            outside_part[k] = v[~is_in_partition]

        model.properties = inside_part

        return model, outside_part, is_in_partition
