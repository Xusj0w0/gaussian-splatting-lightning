import os
import os.path as osp
import sys
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from arguments import ModelParams
from external.partition_utils import (VastGSPartitionCoordinates,
                                      VastGSProgressiveDataPartitioning,
                                      focal2fov)
from scene.dataset_readers import (CameraInfo, SceneInfo, fetchPly,
                                   getNerfppNorm, storePly)

import internal.utils.colmap as colmap_utils
from utils.camera_utils import cameraList_from_camInfos_partition
from utils.graphics_utils import BasicPointCloud


def read_scene(dataset_path: str, ply_path: str, extra_trans: Optional[np.ndarray] = None):
    """
    Replace `partition()` in `scene.dataset_readers`, using `colmap_utils` from `internal.utils`.
    `dataset_path/sparse/` should contains *.bin or *.txt files.
    """
    colmap_model: Tuple[
        Dict[int, colmap_utils.Camera], Dict[int, colmap_utils.Image], Dict[int, colmap_utils.Point3D]
    ] = colmap_utils.read_model(os.path.join(dataset_path, "sparse"))
    cameras, images, points3D = colmap_model

    cam_infos = []
    for image_id, image in images.items():
        camera_id = image.camera_id
        camera = cameras[camera_id]
        uid = camera.id
        height, width = camera.height, camera.width

        dtype = image.tvec.dtype
        if extra_trans is not None:
            w2c = np.eye(4, dtype=dtype)
            w2c[:3, :3], w2c[:3, -1] = image.qvec2rotmat(), image.tvec
            w2c_extra = w2c @ np.linalg.inv(extra_trans)
            R, T = w2c_extra[:3, :3].transpose(), w2c_extra[:3, -1]
        else:
            R, T = image.qvec2rotmat().transpose(), image.tvec

        if camera.model == "SIMPLE_PINHOLE":
            focal_length_x = camera.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif camera.model == "PINHOLE":
            focal_length_x = camera.params[0]
            focal_length_y = camera.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        cam_infos.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=None,
                image_path=None,
                image_name=image.name,
                width=width,
                height=height,
            )
        )
    # cam_infos = sorted(cam_infos, key=lambda x: x.image_name)  # affect *-based_assignments
    nerf_normalization = getNerfppNorm(cam_infos)

    xyz, rgb = extract_from_points3D(points3D)
    # ply_path = osp.join(output_path, "input.ply")
    if not osp.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path, man_trans=extra_trans)  # transform applied pcd

    dist_threshold = 99
    points, colors, normals = pcd.points, pcd.colors, pcd.normals
    points_threshold = np.percentile(points[:, 1], dist_threshold)  # use dist_ratio to exclude outliers

    colors = colors[points[:, 1] < points_threshold]
    normals = normals[points[:, 1] < points_threshold]
    points = points[points[:, 1] < points_threshold]
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

    return pcd, cam_infos, nerf_normalization


def data_partition(
    dataset_path: str,
    output_path: str,
    partition_path: str,
    regions: List[int] = [2, 4],
    extra_trans: Optional[str] = "",
):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(partition_path, exist_ok=True)
    if len(extra_trans) > 0:
        try:
            seq = [float(s) for s in extra_trans.strip().split()]
        except:
            raise ValueError("Invalid extra_transformation.")
        assert len(seq) == 12, "Invalid extra_transformation."
        extra_trans = np.array(seq).reshape(3, 4)
    else:
        extra_trans = None

    ply_path = osp.join(output_path, "input.ply")
    pcd, cam_infos, nerf_normalization = read_scene(
        dataset_path=dataset_path, ply_path=ply_path, extra_trans=extra_trans
    )
    all_cameras = cameraList_from_camInfos_partition(cam_infos, Namespace(data_device="cpu"))
    DataPartitioning = VastGSProgressiveDataPartitioning(
        pcd=pcd,
        train_cameras=all_cameras,
        model_path=output_path,
        m_region=regions[0],
        n_region=regions[1],
        extend_rate=0.2,
        visible_rate=0.25,
        extra_trans=extra_trans,
    )
    partition_result = DataPartitioning.partition_scene

    # client = 0
    # partition_id_list = []
    # for partition in partition_result:
    #     partition_id_list.append(partition.partition_id)
    #     camera_infos = partition.cameras
    #     image_name_list = [camera_infos[i].camera.image_name + ".jpg" for i in range(len(camera_infos))]
    #     txt_file = f"{output_path}/partition_point_cloud/visible/{partition.partition_id}_camera.txt"
    #     # 打开一个文件用于写入，如果文件不存在则会被创建
    #     with open(txt_file, "w") as file:
    #         # 遍历列表中的每个元素
    #         for item in image_name_list:
    #             # 将每个元素写入文件，每个元素占一行
    #             file.write(f"{item}\n")
    #     client += 1

    # prepare necessary files for gspl-lightning partition training
    # 1. camera list for each partition
    # 2. pcd for initialization
    # 3. partitions.pt
    # camera list
    os.makedirs(osp.join(partition_path, "image_lists"), exist_ok=True)
    for partition in partition_result:
        camera_infos = partition.cameras
        txt_file = f"{partition_path}/image_lists/{partition.partition_id}.txt"
        with open(txt_file, "w") as file:
            for camera_info in camera_infos:
                file.write(f"{camera_info.camera.image_name}\n")

    # pcd
    os.makedirs(osp.join(partition_path, "init_pcds"), exist_ok=True)
    DataPartitioning.save_reverted_pcds(osp.join(partition_path, "init_pcds"))

    # partitions.py
    partitioning_info = {}

    partitioning_info["scene_config"] = {}

    scene_bbox = torch.tensor([p.extend_point_bbox for p in partition_result])
    min_xz = torch.min(scene_bbox[:, 0::2], dim=0)[0][[0, 2]]  # [0] for max_value, [[0, 2]] to index x&z
    max_xz = torch.max(scene_bbox[:, 1::2], dim=0)[0][[0, 2]]
    partitioning_info["scene_bounding_box"] = {
        "bounding_box": {"min": min_xz, "max": max_xz},
        "n_partitions": torch.tensor(regions),
        "origin_partition_offset": None,
    }

    id_tensor = torch.tensor([[int(s) for s in p.partition_id.split("_")] for p in partition_result])
    xy_tensor = torch.tensor([p.ori_camera_bbox for p in partition_result])
    partitioning_info["partition_coordinates"] = {"id": id_tensor, "xy": xy_tensor}

    partitioning_info["visibilities"] = torch.zeros((len(id_tensor), len(all_cameras)), dtype=torch.float32)
    partitioning_info["location_based_assignments"] = torch.from_numpy(DataPartitioning.location_based_assignments)
    partitioning_info["final_assignments"] = torch.from_numpy(DataPartitioning.final_assignments)
    partitioning_info["extra_data"] = {
        "up": torch.from_numpy(np.linalg.inv(extra_trans)[:3, 1]),
        "rotation_transform": torch.from_numpy(extra_trans),
    }
    torch.save(partitioning_info, osp.join(partition_path, "partitions.pt"))


def extract_from_points3D(points3D: Dict[int, colmap_utils.Point3D]):
    xyzs, rgbs = [], []
    for point3D in points3D.values():
        xyzs.append(point3D.xyz)
        rgbs.append(point3D.rgb)
    xyz, rgb = np.stack(xyzs, axis=0), np.stack(rgbs, axis=0)
    return xyz, rgb


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str, default="")
    parser.add_argument("--output_path", required=True, type=str, default="")
    parser.add_argument("--partitions_path", required=True, type=str, default="")
    parser.add_argument("--regions", required=True, type=str, default="2,4")
    parser.add_argument("--extra_trans", type=str, default="")
    return parser


def main(args):
    data_partition(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        partition_path=args.partitions_path,
        regions=args.regions,
        extra_trans=args.extra_trans,
    )


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.regions = [int(s) for s in args.regions.split(",")]
    main(args)

    # dataset_path = "/data/xusj/Projects/3drec/gaussian-splatting-lightning/datasets/Mill-19/rubble/colmap"
    # output_path = "/data/xusj/Projects/3drec/gaussian-splatting-lightning/tmp/rubble/vastgs_output"
    # partitions_path = "/data/xusj/Projects/3drec/gaussian-splatting-lightning/tmp/rubble/partitions"
    # regions = [3, 3]
    # st = """
    #     0.931001 -0.060253 0.360009 -32.820862
    #     0.054672 0.998174 0.025674 -119.222771
    #     -0.360898 -0.004220 0.932596 -9.532154
    #     0.000000 0.000000 0.000000 1.000000
    # """
    # extra_trans = np.array([[float(s) for s in line.strip().split()] for line in st.strip().splitlines()])
    # data_partition(
    #     dataset_path=dataset_path,
    #     output_path=output_path,
    #     partition_path=partitions_path,
    #     regions=regions,
    #     extra_trans=extra_trans,
    # )
