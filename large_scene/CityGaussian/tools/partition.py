import json
import os
import os.path as osp
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from internal.configs.dataset import DatasetParams

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
import numpy as np
import torch
import yaml
from arguments import GroupParams
from scene import LargeScene
# from scene.gaussian_model import GaussianModel
from tqdm import tqdm

from internal.dataparsers.colmap_dataparser import Colmap
from internal.dataparsers.dataparser import DataParserOutputs
from internal.utils.gaussian_model_loader import GaussianModelLoader

# from transforms3d.quaternions import mat2quat

# from utils.camera_utils import loadCam_woImage
# from utils.general_utils import parse_cfg, safe_state
# from utils.large_utils import contract_to_unisphere, get_default_aabb
# from utils.loss_utils import ssim


def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--partition_training_path",
        default="outputs/citygs-rubble_copy",
        # required=True,
        type=str,
        help="Path to coarse training ckpt.",
    )
    parser.add_argument(
        "--regions",
        required=True,
        type=str,
        default="2,4",
        help="Split number along x-, z- and y-axis, like '2,4,2'. \
            If only two dimensions are specified, split number along y-axis will be 1.",
    )
    parser.add_argument("--manhattan_trans", type=str, default="", help="Relative path to dataset_path")

    return parser


def parse_manhattan_trans(dataset_path: str, manhattan_trans: str):
    if len(manhattan_trans) > 0:
        manhattan_path = osp.join(dataset_path, manhattan_trans)
        assert osp.exists(manhattan_path), "File to manhattan transformation not found."
        try:
            with open(manhattan_path, "r") as f:
                trans_mat = " ".join([l.strip() for l in f.readlines()])
            seq = [float(s) for s in trans_mat.split()]
        except:
            raise ValueError("Parse manhattan transformation failed.")
        assert len(seq) == 16, "Invalid manhattan transformation."
        manhattan_trans = np.array(seq).reshape(4, 4)
    else:
        manhattan_trans = None
    return manhattan_trans


def parse_colmap_data(dataset_path, dataparser_config_dict):
    # modify config dict, thus train_cameras contains all cameras including those for validation
    dataparser_config_dict.update(
        {
            "image_list": None,
            "split_mode": "reconstruction",
            "points_from": "random",  # set `points_from` to 'random' to avoid loading points3D
        }
    )
    dataparser_config = Colmap(**dataparser_config_dict)
    dataparser_outputs: DataParserOutputs = dataparser_config.instantiate(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()
    return dataparser_outputs


def block_partitioning(
    partition_training_path: str, output_path: str, regions: List[int] = [2, 4, 1], manhattan_trans: str = ""
):
    # load model from ckpt
    ckpt_path = GaussianModelLoader.search_load_file(osp.join(partition_training_path, "coarse"))
    ckpt = torch.load(ckpt_path)
    coarse_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, "cpu")
    dataset_path = ckpt["datamodule_hyper_parameters"]["path"]

    # parse data from colmap
    dataparser_config_dict = asdict(ckpt["datamodule_hyper_parameters"]["parser"])
    dataparser_output = parse_colmap_data(dataset_path=dataset_path, dataparser_config_dict=dataparser_config_dict)

    # parse manhattan transformation
    manhattan_trans = parse_manhattan_trans(dataset_path=dataset_path, manhattan_trans=manhattan_trans)

    camera_center = dataparser_output.train_set.cameras.camera_center
    


def main():
    args = make_parser().parse_args()
    regions = args.regions.split(",")
    if len(regions) == 2:
        regions.append("1")
    regions = [int(i.strip()) for i in regions]
    assert len(regions) == 3, "Dimension of regions should be 3."
    args.regions = regions


if __name__ == "__main__":
    main()
