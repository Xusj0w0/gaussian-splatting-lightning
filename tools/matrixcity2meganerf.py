# modified from notebook/matrixcity2meganerf.ipynb
# convert MatrixCity dataset to MegaNeRF format
# images in *.tar should be extracted to <corresponding folder>/rbg/ before running this script

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import viser.transforms as vt

from internal.dataparsers.dataparser import DataParserOutputs
from internal.dataparsers.matrix_city_dataparser import MatrixCity

ROTATION_HALF_PI = torch.eye(4, dtype=torch.double)
ROTATION_HALF_PI[:3, :3] = torch.from_numpy(vt.SO3.from_y_radians(-np.pi / 2).as_matrix())


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output_path", type=str, required=True)
    return parser


def parse_args(parser):
    args = parser.parse_args()

    return args


def parse_data(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    train_root = dataset_path / "train"
    train_ls = [str(p.relative_to(dataset_path)) for p in train_root.rglob("*transforms.json")]
    test_root = dataset_path / "test"
    test_ls = [str(p.relative_to(dataset_path)) for p in test_root.rglob("*transforms.json")]
    dataparser = MatrixCity(
        train=train_ls,
        test=test_ls,
    ).instantiate(dataset_path, output_path, 0)
    train_set, _ = dataparser._parse_json(dataparser.params.train, False)
    test_set, _ = dataparser._parse_json(dataparser.params.test, False)
    dataparser_outputs = DataParserOutputs(
        train_set=train_set,
        val_set=test_set,
        test_set=test_set,
        point_cloud=None,
        appearance_group_ids=None,
    )
    return dataparser, dataparser_outputs


def rotate_cameras(dataparser_outputs: DataParserOutputs):
    train_c2w_from_dataparser = torch.linalg.inv(
        dataparser_outputs.train_set.cameras.world_to_camera.transpose(1, 2).to(torch.double)
    )
    train_c2w_from_dataparser[:, :3, 1:3] *= -1
    rotated_train_c2ws = ROTATION_HALF_PI @ train_c2w_from_dataparser
    test_c2w_from_dataparser = torch.linalg.inv(
        dataparser_outputs.test_set.cameras.world_to_camera.transpose(1, 2).to(torch.double)
    )
    test_c2w_from_dataparser[:, :3, 1:3] *= -1
    rotated_test_c2ws = ROTATION_HALF_PI @ test_c2w_from_dataparser
    return rotated_train_c2ws, rotated_test_c2ws


def recenter_and_scale_cameras(rotated_train_c2ws, rotated_test_c2ws):
    camera_centers = rotated_train_c2ws[:, :3, 3]
    origin = (torch.max(camera_centers, dim=0).values + torch.min(camera_centers, dim=0).values) * 0.5

    camera_centers_moved = camera_centers - origin[None, :]
    scale = torch.max(camera_centers_moved)

    recentered_and_scaled_train_c2ws = torch.clone(rotated_train_c2ws)
    recentered_and_scaled_train_c2ws[:, :3, 3] -= origin[None, :]
    recentered_and_scaled_train_c2ws[:, :3, 3] /= scale

    recentered_and_scaled_test_c2ws = torch.clone(rotated_test_c2ws)
    recentered_and_scaled_test_c2ws[:, :3, 3] -= origin[None, :]
    recentered_and_scaled_test_c2ws[:, :3, 3] /= scale

    return (
        recentered_and_scaled_train_c2ws,
        recentered_and_scaled_test_c2ws,
        origin,
        scale,
    )


def save_image_set(output_path, target_image_set, target_c2ws, target_split, idx_offset: int):
    rgb_dir = os.path.join(output_path, target_split, "rgbs")
    metadata_dir = os.path.join(output_path, target_split, "metadata")

    for i in os.scandir(rgb_dir):
        if not i.is_dir(follow_symlinks=False):
            os.unlink(i.path)
    for i in os.scandir(metadata_dir):
        if not i.is_dir(follow_symlinks=False):
            os.unlink(i.path)

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    for idx in range(len(target_image_set)):
        name_idx = idx + idx_offset
        os.link(target_image_set.image_paths[idx], os.path.join(rgb_dir, "{:06d}.png".format(name_idx)))
        torch.save(
            {
                "H": target_image_set.cameras.height[idx].int().item(),
                "W": target_image_set.cameras.width[idx].int().item(),
                "c2w": target_c2ws[idx].to(torch.float)[:3],
                "intrinsics": torch.tensor(
                    [
                        target_image_set.cameras.fx[idx],
                        target_image_set.cameras.fy[idx],
                        target_image_set.cameras.cx[idx],
                        target_image_set.cameras.cy[idx],
                    ]
                ),
            },
            os.path.join(metadata_dir, "{:06d}.pt".format(name_idx)),
        )


if __name__ == "__main__":
    parser = make_parser()
    args = parse_args(parser)

    dataset_path = args.path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    dataparser, dataparser_outputs = parse_data(dataset_path, output_path)
    rotated_train_c2ws, rotated_test_c2ws = rotate_cameras(dataparser_outputs)
    (
        recentered_and_scaled_train_c2ws,
        recentered_and_scaled_test_c2ws,
        origin,
        scale,
    ) = recenter_and_scale_cameras(rotated_train_c2ws, rotated_test_c2ws)
    camera_center_extent = (
        torch.max(recentered_and_scaled_train_c2ws[:, :3, 3], dim=0).values
        - torch.min(recentered_and_scaled_train_c2ws[:, :3, 3], dim=0).values
    )

    os.makedirs(output_path, exist_ok=True)
    torch.save(
        {
            "origin_drb": origin,
            "pose_scale_factor": scale.item(),
        },
        os.path.join(output_path, "coordinates.pt"),
    )

    for split in ["train", "val"]:
        for d in ["rgbs", "metadata"]:
            os.makedirs(os.path.join(output_path, split, d), exist_ok=True)
    save_image_set(
        output_path,
        dataparser_outputs.train_set,
        recentered_and_scaled_train_c2ws,
        "train",
        0,
    )
    save_image_set(
        output_path,
        dataparser_outputs.test_set,
        recentered_and_scaled_test_c2ws,
        "val",
        len(dataparser_outputs.train_set),
    )
