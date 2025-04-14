import argparse
import os
import os.path as osp
import shutil
from typing import Dict, List, Tuple

import internal.utils.colmap as colmap_utils


def make_parser():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test sets, including images (defined by down sample factor) and sparse model"
    )
    parser.add_argument("input_dir", type=str, help="input directory")
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--val_image_list", type=str, help="path to text validation image list")
    parser.add_argument("--down_sample_factor", type=int, default=1, help="down sample factor")
    parser.add_argument(
        "--skip_sparse",
        action="store_true",
        help="skip sparse model. if sparse model have split previously, enable this option to only split images",
    )
    parser.add_argument("--use_symlink", action="store_true", help="use symlink instead of copy images", default=True)
    return parser


def split_sparse_model(args):
    sparse_model_path = osp.join(args.input_dir, "sparse")
    if not (
        osp.exists(osp.join(sparse_model_path, "images.bin")) or osp.exists(osp.join(sparse_model_path, "images.txt"))
    ):
        sparse_model_path = osp.join(sparse_model_path, "0")
    if not (
        osp.exists(osp.join(sparse_model_path, "images.bin")) or osp.exists(osp.join(sparse_model_path, "images.txt"))
    ):
        raise ValueError("Sparse model not found in {}".format(sparse_model_path))

    cameras: Dict[int, colmap_utils.Camera]; images: Dict[int, colmap_utils.Image]; points3D: Dict[int, colmap_utils.Point3D] # fmt: skip
    cameras, images, points3D = colmap_utils.read_model(sparse_model_path)

    with open(args.val_image_list, "r") as f:
        val_image_names = [line.strip() for line in f.readlines()]

    val_image_ids = []
    for image_id, image in images.items():
        if image.name in val_image_names:
            val_image_ids.append(image_id)

    # split images
    train_image_ids = list(set(images.keys()).difference(set(val_image_ids)))
    train_images = {i: images[i] for i in train_image_ids}
    val_images = {i: images[i] for i in val_image_ids}

    # split cameras
    train_camera_ids = {image.camera_id for image in train_images.values()}
    val_camera_ids = {image.camera_id for image in val_images.values()}
    train_cameras = {i: cameras[i] for i in train_camera_ids}
    val_cameras = {i: cameras[i] for i in val_camera_ids}

    # split point cloud: only images and cameras for validation, point cloud are for initialization in training
    train_points3D = points3D
    val_points3D = {}

    train_sparse_model_path = osp.join(args.output_dir, "train", osp.relpath(sparse_model_path, args.input_dir))
    os.makedirs(train_sparse_model_path, exist_ok=True)
    colmap_utils.write_model(train_cameras, train_images, train_points3D, train_sparse_model_path)

    val_sparse_model_path = osp.join(args.output_dir, "val", osp.relpath(sparse_model_path, args.input_dir))
    os.makedirs(val_sparse_model_path, exist_ok=True)
    colmap_utils.write_model(val_cameras, val_images, val_points3D, val_sparse_model_path)


def split_images(args):
    image_dirname = "images" + (f"_{int(args.down_sample_factor)}" if args.down_sample_factor != 1 else "")
    downsampled_image_dir = osp.join(args.input_dir, image_dirname)
    assert osp.exists(downsampled_image_dir), "Downsampled images not found"

    with open(args.val_image_list, "r") as f:
        val_image_names = [line.strip() for line in f.readlines()]

    for val_image_name in val_image_names:
        val_image_path = osp.join(downsampled_image_dir, val_image_name)
        assert osp.exists(val_image_path), "Validation image not found"
        dst_image_path = osp.join(args.output_dir, "val", image_dirname, val_image_name)
        os.makedirs(osp.dirname(dst_image_path), exist_ok=True)
        if args.use_symlink:
            os.symlink(val_image_path, dst_image_path)
        else:
            shutil.copyfile(val_image_path, dst_image_path)

    train_image_names = list(set(os.listdir(downsampled_image_dir)).difference(set(val_image_names)))
    for train_image_name in train_image_names:
        train_image_path = osp.join(downsampled_image_dir, train_image_name)
        assert osp.exists(train_image_path), "Training image not found"
        dst_image_path = osp.join(args.output_dir, "train", image_dirname, train_image_name)
        os.makedirs(osp.dirname(dst_image_path), exist_ok=True)
        if args.use_symlink:
            os.symlink(train_image_path, dst_image_path)
        else:
            shutil.copyfile(train_image_path, dst_image_path)


def split_dataset(args):
    # make sure split directory exists
    assert osp.exists(args.input_dir), "Input directory not found"
    assert osp.exists(args.val_image_list), "Validation image list text file not found"

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.skip_sparse:
        split_sparse_model(args)
    split_images(args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    split_dataset(args)

    # debug configuration
    # {
    #     "name": "split_dataset",
    #     "type": "debugpy",
    #     "request": "launch",
    #     "program": "tools/split_dataset.py",
    #     "console": "integratedTerminal",
    #     "args": [
    #         "datasets/MegaNeRF/rubble/colmap",
    #         "--output_dir=/data/xusj/Projects/3drec/try/Momentum-GS/data/rubble_unsplit",
    #         "--val_image_list=datasets/MegaNeRF/rubble/colmap/splits/val_images.txt"
    #     ]
    # },
