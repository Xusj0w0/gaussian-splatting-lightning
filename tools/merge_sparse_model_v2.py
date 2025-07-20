# modified for DOGS (sort images by name)
import argparse
import os
import os.path as osp
import shutil
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

import internal.utils.colmap as colmap_utils
from internal.utils.colmap import Camera, Image, Point3D


def make_parser():
    parser = argparse.ArgumentParser(description="Merge train/val sparse model.")
    parser.add_argument("input_dir", type=str, help="input directory")
    parser.add_argument("--output_dir", type=str, help="output directory")
    return parser


def is_same_camera(camera: Camera, camera_ref: Camera):
    if camera.model != camera_ref.model:
        return False
    if camera.width != camera_ref.width or camera.height != camera_ref.height:
        return False
    if not np.allclose(camera.params, camera_ref.params, atol=1e-3):
        return False
    return True


def merge_sparse_model(args):
    sparse_model_paths = []
    for split in ["train", "val"]:
        sparse_model_path = osp.join(args.input_dir, split, "sparse")
        if not (
            osp.exists(osp.join(sparse_model_path, "images.bin"))
            or osp.exists(osp.join(sparse_model_path, "images.txt"))
        ):
            sparse_model_path = osp.join(sparse_model_path, "0")
        if not (
            osp.exists(osp.join(sparse_model_path, "images.bin"))
            or osp.exists(osp.join(sparse_model_path, "images.txt"))
        ):
            raise ValueError("Sparse model not found in {}".format(sparse_model_path))
        sparse_model_paths.append(sparse_model_path)
    train_sparse_path, val_sparse_path = sparse_model_paths

    train_cameras: Dict[int, Camera]; train_images: Dict[int, Image]; train_points3d: Dict[int, Point3D] # fmt: skip
    train_cameras, train_images, train_points3d = colmap_utils.read_model(train_sparse_path)
    val_cameras: Dict[int, Camera]; val_images: Dict[int, Image]; val_points3d: Dict[int, Point3D] # fmt: skip
    val_cameras, val_images, val_points3d = colmap_utils.read_model(val_sparse_path)

    image_name_to_id = {}
    for image_id, image in train_images.items():
        image_name_to_id[image.name] = ("train", image_id)
    for image_id, image in val_images.items():
        image_name_to_id[image.name] = ("val", image_id)
    all_image_names = sorted(
        [image.name for image in train_images.values()] + [image.name for image in val_images.values()]
    )

    new_cameras = deepcopy(train_cameras)

    # merge images
    new_images = {}
    train_image_mapping, val_image_mapping = {}, {}
    image_id_to_split_and_old_id = {}
    for idx, image_name in enumerate(all_image_names):
        split, image_id = image_name_to_id[image_name]
        image_id_to_split_and_old_id[idx + 1] = (split, image_id)
        if split == "train":
            target_image = train_images[image_id]
            train_image_mapping[image_id] = idx + 1
        else:
            target_image = val_images[image_id]
            val_image_mapping[image_id] = idx + 1

        new_image = Image(  # "id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"
            id=idx + 1,  # start from 1
            qvec=target_image.qvec,
            tvec=target_image.tvec,
            camera_id=1,
            name=image_name,
            xys=target_image.xys,
            point3D_ids=image.point3D_ids,  # pts3d not merged yet
        )
        new_images[idx + 1] = new_image

    # merge points3d
    new_points3d = {}
    train_pts3d_mapping, val_pts3d_mapping = {}, {}
    num_points3d = 0
    for point3d_id, point3d in train_points3d.items():
        num_points3d += 1
        image_ids = np.array([train_image_mapping[i] for i in point3d.image_ids.tolist()])
        image_ids = image_ids.astype(point3d.image_ids.dtype)
        new_point = Point3D(
            id=num_points3d,
            xyz=point3d.xyz,
            rgb=point3d.rgb,
            error=point3d.error,
            image_ids=image_ids,
            point2D_idxs=point3d.point2D_idxs,
        )
        new_points3d[num_points3d] = new_point
        train_pts3d_mapping[point3d_id] = num_points3d
    for point3d_id, point3d in val_points3d.items():
        num_points3d += 1
        image_ids = np.array([val_image_mapping[i] for i in point3d.image_ids.tolist()])
        image_ids = image_ids.astype(point3d.image_ids.dtype)
        new_point = Point3D(
            id=num_points3d,
            xyz=point3d.xyz,
            rgb=point3d.rgb,
            error=point3d.error,
            image_ids=image_ids,
            point2D_idxs=point3d.point2D_idxs,
        )
        new_points3d[num_points3d] = new_point
        val_pts3d_mapping[point3d_id] = num_points3d

    # update point3D_ids in images
    for new_img_id, image in new_images.items():
        split, old_img_id = image_id_to_split_and_old_id[new_img_id]

        if split == "train":
            old_image = train_images[old_img_id]
            mapping = train_pts3d_mapping
        else:
            old_image = val_images[old_img_id]
            mapping = val_pts3d_mapping

        points3d_ids = [mapping[i] if i > 0 and i in mapping else -1 for i in old_image.point3D_ids.tolist()]
        new_images[new_img_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=np.array(points3d_ids).astype(image.point3D_ids.dtype),
        )

    return new_cameras, new_images, new_points3d


def main():
    args = make_parser().parse_args()
    if args.output_dir is None or len(args.output_dir) == 0:
        args.output_dir = osp.join(args.input_dir, "sparse")
    os.makedirs(args.output_dir, exist_ok=True)

    cameras, images, points3d = merge_sparse_model(args)
    colmap_utils.write_model(cameras, images, points3d, args.output_dir)


if __name__ == "__main__":
    main()
