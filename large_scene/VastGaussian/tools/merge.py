# modified from vastgs seamless_merging and utils/merge_partitions_v2.py

import gc
import json
import os
import os.path as osp
import re
import sys
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
import numpy as np
import torch
from external.partition_utils import VastGSPartitionCoordinates
from tqdm.auto import tqdm

from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.appearance_feature_gaussian import \
    AppearanceFeatureGaussianModel
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_utils import (GaussianPlyUtils,
                                           GaussianTransformUtils)
from internal.utils.partitioning_utils import MinMaxBoundingBox

MERGABLE_PROPERTY_NAMES = ["means", "shs_dc", "shs_rest", "scales", "rotations", "opacities"]


def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--partition_training_path", type=str, required=True, help="Partition training result directory."
    )
    parser.add_argument(
        "--partition_path",
        type=str,
        required=True,
        help="Partitioning info saving directory. Containing files partitions.pt, etc.",
    )
    return parser


def find_ckpt_path(partition_res_dir: str, partition_id: str) -> VanillaGaussianModel:
    partition_path = osp.join(partition_res_dir, partition_id)
    assert osp.exists(partition_path), "Partition path not exist."
    assert osp.exists(osp.join(partition_res_dir, f"{partition_id}-trained")), "Partition training not finished yet."

    # find ckpt
    ckpt_dir = osp.join(partition_path, "checkpoints")
    max_step, ckpt_path = 0, None
    for ckpt_file in Path(ckpt_dir).glob("*.ckpt"):
        match = re.search(r"step=(\d+)", ckpt_file.stem)
        step = int(match.group(1)) if match else 0
        if step > max_step:
            ckpt_path = ckpt_file
            max_step = step

    return str(ckpt_path)


def convert_load_transform_ply(
    partition_res_dir: str, partition_id: str, manhattan_trans: Optional[np.ndarray] = None
) -> GaussianPlyUtils:
    # TODO: support various gaussian types
    partition_path = osp.join(partition_res_dir, partition_id)
    assert osp.exists(partition_path), "Partition path not exist."
    assert osp.exists(osp.join(partition_res_dir, f"{partition_id}-trained")), "Partition training not finished yet."

    # find ckpt
    ckpt_dir = osp.join(partition_path, "checkpoints")
    max_step, ckpt_path = 0, None
    for ckpt_file in Path(ckpt_dir).glob("*.ckpt"):
        match = re.search(r"step=(\d+)", ckpt_file.stem)
        step = int(match.group(1)) if match else 0
        if step > max_step:
            ckpt_path = ckpt_file
            max_step = step

    # convert and load ply
    ply_file = Path(partition_res_dir) / "point_cloud" / "iteration_{}".format(max_step) / "point_cloud.ply"
    if not ply_file.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        GaussianPlyUtils.load_from_state_dict(ckpt["state_dict"]).to_ply_format().save_to_ply(str(ply_file) + ".tmp")
        os.rename(str(ply_file) + ".tmp", str(ply_file))
    model: GaussianPlyUtils = GaussianPlyUtils.load_from_ply(str(ply_file))

    # maybe transform
    if manhattan_trans is not None:
        assert manhattan_trans.shape == (4, 4), "Manhattan transform should be 4x4 matrix."
        xyz, rotations, features_dc, features_rest = (
            model.xyz,
            model.rotations,
            model.features_dc,
            model.features_rest,
        )
        _xyz, _rotations = GaussianTransformUtils.rotate_by_matrix(xyz, rotations, manhattan_trans[:3, :3])
        features = torch.cat([features_dc, features_rest], dim=1)  # [N, sh_degree, 3]
        _features = GaussianTransformUtils.transform_shs(features, manhattan_trans[:3, :3])
        _features_dc, _features_rest = _features[:, :1], _features[:, 1:]
        model = GaussianPlyUtils(
            sh_degrees=model.sh_degrees,
            xyz=_xyz,
            opacities=model.opacities,
            features_dc=_features_dc,
            features_rest=_features_rest,
            scales=model.scales,
            rotations=_rotations,
        )

    return model


def extend_inf_x_z_bbox(partition_id, m_region, n_region):
    x, z = int(partition_id.split("_")[0]), int(partition_id.split("_")[1])  # 获取块所在编号
    flag = [False] * 4
    if x == 1:
        flag[0] = True
    if z == 1:
        flag[1] = True
    if x == m_region:
        flag[2] = True
    if z == n_region:
        flag[3] = True
    return flag


def get_partition_gaussian_mask(
    means: torch.Tensor, bounding_box: MinMaxBoundingBox, manhattan_transform: Optional[torch.Tensor] = None
):
    if manhattan_transform is not None:
        manhattan_transform = manhattan_transform.to(means)
        means = means @ manhattan_transform[:3, :3].T + manhattan_transform[:3, -1]
    is_ge_min = torch.prod(torch.ge(means[..., 0::2], bounding_box.min), dim=-1)
    is_lt_max = torch.prod(torch.lt(means[..., 0::2], bounding_box.max), dim=-1)
    is_in_bounding_box = torch.logical_and(is_ge_min, is_lt_max)
    return is_in_bounding_box


def merge(partition_path: str, partition_training_path: str):
    sys.path.insert(0, osp.join(os.getcwd(), "utils"))
    from utils.merge_partitions_v2 import (fuse_appearance_features,
                                           fuse_mip_filters, update_ckpt)

    partitioning_info = torch.load(osp.join(partition_path, "partitions.pt"))
    m_region, n_region = partitioning_info["scene_bounding_box"]["n_partitions"].tolist()
    manhattan_transform = partitioning_info.get("extra_data", {}).get("rotation_transform", None)
    dataset_path = partitioning_info.get("extra_data", {}).get("dataset_path", None)
    assert dataset_path is not None, "Dataset path not found in partitioning info."
    coordinates = VastGSPartitionCoordinates(**partitioning_info["partition_coordinates"])
    bboxes = coordinates.get_bounding_boxes()

    gaussian_to_merge = {}
    image_name_to_camera = None
    with tqdm(range(len(coordinates.id)), desc="Pre-processing") as t:
        for pid in t:
            partition_id = coordinates.get_str_id(pid)
            t.set_description("{}".format(partition_id))
            t.set_postfix_str("Loading checkpoint...")

            # load ckpt
            # ckpt_path = find_ckpt_path(partition_training_path, partition_id)
            ckpt_path = GaussianModelLoader.search_load_file(osp.join(partition_training_path, partition_id))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            partition_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, "cpu")

            # crop partition model
            t.set_postfix_str("Splitting...")
            bbox = bboxes[pid]
            flag = extend_inf_x_z_bbox(partition_id, m_region, n_region)
            modified_bbox = VastGSPartitionCoordinates.modify(bbox, flag)
            mask = get_partition_gaussian_mask(partition_model.means, modified_bbox, manhattan_transform)
            # xmin, xmax, zmin, zmax = modify_bbox([xmin, xmax, zmin, zmax], flag)
            # mask = extract_point_cloud(partition_model.xyz, [xmin, xmax, -math.inf, math.inf, zmin, zmax])
            inside_part = {}
            for k, v in partition_model.properties.items():
                inside_part[k] = v[mask]
            partition_model.properties = inside_part

            # maybe convert to VanillaGaussianModel
            if isinstance(partition_model, AppearanceFeatureGaussianModel):
                with open(osp.join(osp.dirname(osp.dirname(ckpt_path)), "cameras.json"), "r") as f:
                    cameras_json = json.load(f)

                if image_name_to_camera is None:
                    dataparser_config = Colmap(
                        split_mode="reconstruction",  # load all cameras
                    )
                    for i in [
                        "image_dir",
                        "mask_dir",
                        "scene_scale",
                        "reorient",
                        "appearance_groups",
                        "down_sample_factor",
                        "down_sample_rounding_mode",
                    ]:
                        setattr(dataparser_config, i, getattr(ckpt["datamodule_hyper_parameters"]["parser"], i))

                    dataparser_outputs = dataparser_config.instantiate(
                        path=dataset_path,
                        output_path=os.getcwd(),
                        global_rank=0,
                    ).get_outputs()

                    image_name_to_camera = {}
                    for idx in range(len(dataparser_outputs.train_set)):
                        image_name = dataparser_outputs.train_set.image_names[idx]
                        camera = dataparser_outputs.train_set.cameras[idx]
                        image_name_to_camera[image_name] = camera

                t.set_postfix_str("Fusing...")
                fuse_appearance_features(
                    ckpt,
                    partition_model,
                    cameras_json,
                    image_name_to_camera=image_name_to_camera,
                )

            if isinstance(partition_model, MipSplattingModelMixin):
                t.set_postfix_str("Fusing MipSplatting filters...")
                fuse_mip_filters(partition_model)

            for property in MERGABLE_PROPERTY_NAMES:
                gaussian_to_merge.setdefault(property, []).append(partition_model.get_property(property))

    merged_gaussians = {}
    for k, v in gaussian_to_merge.items():
        merged_gaussians[k] = torch.cat(v, dim=0)
        v.clear(),
        gc.collect()
        torch.cuda.empty_cache()

    update_ckpt(ckpt, merged_gaussians, partition_model.max_sh_degree)
    torch.save(ckpt, osp.join(partition_training_path, "merged.ckpt"))


def main():
    args = make_parser().parse_args()
    merge(
        partition_path=args.partition_path,
        partition_training_path=args.partition_training_path,
    )


if __name__ == "__main__":
    main()
