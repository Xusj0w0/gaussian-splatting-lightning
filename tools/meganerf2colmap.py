# modified from utils/meganerf2colmap.py

import argparse
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import torch

from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.utils import colmap
from internal.utils.graphics_utils import fetch_ply, store_ply


def array_to_blob(array):
    return array.tostring()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--colmap_executable", type=str, default="colmap")
    parser.add_argument("--refine", action="store_true", help="Refine intrinsics and extrinsics")
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    colmap_exe_path = args.colmap_executable

    camera_model = "PINHOLE"
    if args.refine:
        camera_model = "OPENCV"

    coordinates = torch.load(os.path.join(args.path, "coordinates.pt"), map_location="cpu")

    colmap_dir = os.path.join(args.path, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)

    image_metadata_pairs = []
    for split in ["train", "val"]:
        for i in os.scandir(os.path.join(args.path, split, "rgbs")):
            name_without_ext = i.name.split(".")[0]
            image_metadata_pairs.append(
                (
                    i.path,
                    os.path.join(args.path, split, "metadata", "{}.pt".format(name_without_ext)),
                    i.name,
                    split,
                )
            )

    image_dir = os.path.join(colmap_dir, "distorted", "images")
    os.makedirs(image_dir, exist_ok=True)
    for i, _, image_name, split in image_metadata_pairs:
        try:
            os.symlink(os.path.join("..", "..", "..", split, "rgbs", image_name), os.path.join(image_dir, image_name))
        except FileExistsError:
            pass

    colmap_db_path = os.path.join(colmap_dir, "distorted", "colmap.db")
    assert (
        subprocess.call(
            [
                colmap_exe_path,
                "feature_extractor",
                "--database_path",
                colmap_db_path,
                "--image_path",
                image_dir,
                "--ImageReader.camera_model",
                camera_model,
            ]
        )
        == 0
    )

    colmap_db = sqlite3.connect(colmap_db_path)

    def select_image(image_name: str):
        cur = colmap_db.cursor()
        try:
            return cur.execute("SELECT image_id, camera_id FROM images WHERE name = ?", [image_name]).fetchone()
        finally:
            cur.close()

    def set_image_camera_id(image_id: int, camera_id: int):
        cur = colmap_db.cursor()
        try:
            cur.execute("UPDATE images SET camera_id = ? WHERE image_id = ?", [camera_id, image_id])
            colmap_db.commit()
        finally:
            cur.close()

    def update_camera_params(camera_id: int, params: np.ndarray):
        cur = colmap_db.cursor()
        try:
            cur.execute(
                "UPDATE cameras SET params = ? WHERE camera_id = ?",
                [
                    array_to_blob(params),
                    camera_id,
                ],
            )
            colmap_db.commit()
        finally:
            cur.close()

    def delete_unused_cameras():
        cur = colmap_db.cursor()
        try:
            cur.execute("DELETE FROM cameras WHERE camera_id NOT IN (SELECT camera_id FROM images)")
            colmap_db.commit()
        finally:
            cur.close()

    camera_intrinsics_to_camera_id = {}

    images = {}
    cameras = {}
    points = {}

    c2w_transform = torch.tensor(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float,
    ).T
    RDF_TO_DRB_H = torch.tensor(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float,
    )

    image_infos = {}
    image_splits = {"train": [], "val": []}
    for src_path, metadata_path, image_name, split in image_metadata_pairs:
        metadata = torch.load(metadata_path, map_location="cpu")
        image_id, camera_id = select_image(image_name)

        # share intrinsics if possible
        intrinsics_as_dict_key = metadata["intrinsics"]
        intrinsics_as_dict_key = torch.concat(
            [intrinsics_as_dict_key, torch.tensor([metadata["W"], metadata["H"]], dtype=torch.float)],
            dim=-1,
        )
        camera_id = camera_intrinsics_to_camera_id.setdefault(intrinsics_as_dict_key.numpy().tobytes(), camera_id)
        set_image_camera_id(image_id, camera_id)

        image_infos[image_name] = {"image_id": image_id, "camera_id": camera_id, "src_path": src_path, "split": split}
        image_splits[split].append(image_name)

        c2w = torch.eye(4)
        c2w[:3, :] = metadata["c2w"]

        c2w[:3, 3] *= coordinates["pose_scale_factor"]
        c2w[:3, 3] += coordinates["origin_drb"]

        c2w = torch.linalg.inv(RDF_TO_DRB_H) @ c2w @ c2w_transform @ RDF_TO_DRB_H
        w2c = torch.linalg.inv(c2w)

        images[image_id] = colmap.Image(
            image_id,
            qvec=colmap.rotmat2qvec(w2c[:3, :3].numpy()),
            tvec=w2c[:3, 3].numpy(),
            camera_id=camera_id,
            name=image_name,
            xys=np.array([], dtype=np.float64),
            point3D_ids=np.asarray([], dtype=np.int64),
        )

        # a new camera
        if camera_id not in cameras:
            camera_params = metadata["intrinsics"]
            if args.refine:
                camera_params = torch.concat([metadata["intrinsics"], torch.tensor([0.0, 0.0, 0.0, 0.0])], dim=-1)
            update_camera_params(camera_id, camera_params.to(torch.float64).numpy())
            cameras[camera_id] = colmap.Camera(
                id=camera_id,
                model=camera_model,
                width=metadata["W"],
                height=metadata["H"],
                params=camera_params.to(torch.float64).numpy(),
            )

    os.makedirs(os.path.join(colmap_dir, "splits"), exist_ok=True)
    json.dump(
        {"metadata": image_infos, "splits": image_splits},
        open(os.path.join(colmap_dir, "splits", "image_splits.json"), "w"),
        indent=4,
        separators=(", ", ": "),
    )
    with open(os.path.join(colmap_dir, "splits", "train_images.txt"), "w") as f:
        for image_name in image_splits["train"]:
            f.write("{}\n".format(image_name))
    with open(os.path.join(colmap_dir, "splits", "val_images.txt"), "w") as f:
        for image_name in image_splits["val"]:
            f.write("{}\n".format(image_name))

    delete_unused_cameras()

    colmap_db.close()

    sparse_manually_model_dir = os.path.join(colmap_dir, "distorted", "sparse_manually")
    os.makedirs(sparse_manually_model_dir, exist_ok=True)
    colmap.write_images_binary(images, os.path.join(sparse_manually_model_dir, "images.bin"))
    colmap.write_cameras_binary(cameras, os.path.join(sparse_manually_model_dir, "cameras.bin"))
    colmap.write_points3D_binary(points, os.path.join(sparse_manually_model_dir, "points3D.bin"))

    assert (
        subprocess.call(
            [
                colmap_exe_path,
                "exhaustive_matcher",
                "--database_path",
                colmap_db_path,
            ]
        )
        == 0
    )

    sparse_dir_triangulated = os.path.join(colmap_dir, "distorted", "sparse_triangulated")
    if not os.path.exists(sparse_dir_triangulated):
        os.makedirs(sparse_dir_triangulated, exist_ok=True)
        assert (
            subprocess.call(
                [
                    colmap_exe_path,
                    "point_triangulator",
                    "--database_path",
                    colmap_db_path,
                    "--image_path",
                    image_dir,
                    "--input_path",
                    sparse_manually_model_dir,
                    "--output_path",
                    sparse_dir_triangulated,
                    "--Mapper.ba_use_gpu",
                    "1",
                    "--Mapper.ba_gpu_index",
                    args.gpu_id,
                ]
            )
            == 0
        )

    if args.refine:
        # use the intrinsics and extrinsics provided by MegaNeRF will produce a suboptimal result,
        # so run a bundle adjustment to further refine them

        sparse_dir = os.path.join(colmap_dir, "distorted", "sparse")
        if not os.path.exists(sparse_dir):
            os.makedirs(sparse_dir, exist_ok=True)
            assert (
                subprocess.call(
                    [
                        colmap_exe_path,
                        "bundle_adjuster",
                        "--input_path",
                        sparse_dir_triangulated,
                        "--output_path",
                        sparse_dir,
                        "--BundleAdjustment.use_gpu",
                        "1",
                        "--BundleAdjustment.gpu_index",
                        args.gpu_id,
                        "--BundleAdjustment.function_tolerance",
                        "0.000001",
                    ]
                )
                == 0
            )

        dense_dir = os.path.join(colmap_dir, "undistorted")
        if not os.path.exists(dense_dir):
            os.makedirs(dense_dir, exist_ok=True)
            assert (
                subprocess.call(
                    [
                        colmap_exe_path,
                        "image_undistorter",
                        "--image_path",
                        image_dir,
                        "--input_path",
                        sparse_dir,
                        "--output_path",
                        dense_dir,
                    ]
                )
                == 0
            )
        if os.path.exists(os.path.join(dense_dir, "images")) and not os.path.exists(os.path.join(colmap_dir, "images")):
            os.rename(os.path.join(dense_dir, "images"), os.path.join(colmap_dir, "images"))
        if os.path.exists(os.path.join(dense_dir, "sparse")) and not os.path.exists(os.path.join(colmap_dir, "sparse")):
            os.rename(os.path.join(dense_dir, "sparse"), os.path.join(colmap_dir, "sparse"))
    else:
        if os.path.exists(os.path.join(colmap_dir, "distorted", "images")) and not os.path.exists(
            os.path.join(colmap_dir, "images")
        ):
            os.rename(os.path.join(colmap_dir, "distorted", "images"), os.path.join(colmap_dir, "images"))
        if os.path.exists(sparse_dir_triangulated) and not os.path.exists(os.path.join(colmap_dir, "sparse")):
            os.rename(sparse_dir_triangulated, os.path.join(colmap_dir, "sparse"))

    print("Saved to '{}', use this as your dataset path".format(colmap_dir))
    ply_path = os.path.join(colmap_dir, "input.ply")
    if not os.path.exists(ply_path):
        xyz, rgb, _ = ColmapDataParser.read_points3D_binary(os.path.join(colmap_dir, "sparse", "points3D.bin"))
        store_ply(ply_path, xyz, rgb)


def test_main():
    sys.argv = [__file__, os.path.expanduser("~/data/Mega-NeRF/rubble-pixsfm")]
    main()


if __name__ == "__main__":
    main()
