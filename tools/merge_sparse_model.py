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
        if not (osp.exists(osp.join(sparse_model_path, "images.bin")) or osp.exists(osp.join(sparse_model_path, "images.txt"))):
            sparse_model_path = osp.join(sparse_model_path, "0")
        if not (osp.exists(osp.join(sparse_model_path, "images.bin")) or osp.exists(osp.join(sparse_model_path, "images.txt"))):
            raise ValueError("Sparse model not found in {}".format(sparse_model_path))
        sparse_model_paths.append(sparse_model_path)
    train_sparse_path, val_sparse_path = sparse_model_paths

    cameras: Dict[int, Camera]; images: Dict[int, Image]; points3d: Dict[int, Point3D] # fmt: skip
    cameras, images, points3d = colmap_utils.read_model(train_sparse_path)
    val_cams: Dict[int, Camera]; val_imgs: Dict[int, Image]; val_pts3d: Dict[int, Point3D] # fmt: skip
    val_cams, val_imgs, val_pts3d = colmap_utils.read_model(val_sparse_path)

    # merge cameras
    camera_id = max(cameras.keys())
    val_cam_mapping = {}
    for k, v in val_cams.items():
        same_cam_id = -1
        for ref_id, v_ref in cameras.items():
            if is_same_camera(v, v_ref):
                same_cam_id = ref_id
                break
        if same_cam_id < 0:
            camera_id += 1
            new_cam = Camera(id=camera_id, model=v.model, width=v.width, height=v.height, params=v.params)
            cameras[camera_id] = new_cam
            val_cam_mapping[k] = camera_id
        else:
            val_cam_mapping[k] = same_cam_id

    # merge images
    image_id = max(images.keys())
    val_img_mapping = {}
    for k, v in val_imgs.items():  # "id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"
        image_id += 1
        new_img = Image(
            id=image_id,
            qvec=v.qvec,
            tvec=v.tvec,
            camera_id=val_cam_mapping[v.camera_id],
            name=v.name,
            xys=v.xys,
            point3D_ids=np.empty((0,)),  # pts3d not merged yet
        )
        images[image_id] = new_img
        val_img_mapping[k] = image_id

    # merge points3d
    point3d_id = max(points3d.keys())
    val_pts3d_mapping = {}
    for k, v in val_pts3d.items():
        point3d_id += 1
        image_ids = np.array([val_img_mapping[i] for i in v.image_ids.tolist()])
        image_ids = image_ids.astype(v.image_ids.dtype)
        new_point = Point3D(id=point3d_id, xyz=v.xyz, rgb=v.rgb, error=v.error, image_ids=image_ids, point2D_idxs=v.point2D_idxs)
        points3d[point3d_id] = new_point
        val_pts3d_mapping[k] = point3d_id

    # update points3D_ids in images
    for img_id_orig, img_id_mapped in val_img_mapping.items():
        image_orig = val_imgs[img_id_orig]
        image = images[img_id_mapped]
        point3d_ids = np.array([val_pts3d_mapping[i] if i > 0 else -1 for i in image_orig.point3D_ids.tolist()])
        point3d_ids = point3d_ids.astype(image_orig.point3D_ids.dtype)
        images[img_id_mapped] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=point3d_ids,
        )

    return cameras, images, points3d


def main():
    args = make_parser().parse_args()
    if args.output_dir is None or len(args.output_dir) == 0:
        args.output_dir = osp.join(args.input_dir, "sparse")
    os.makedirs(args.output_dir, exist_ok=True)

    cameras, images, points3d = merge_sparse_model(args)
    colmap_utils.write_model(cameras, images, points3d, args.output_dir)


if __name__ == "__main__":
    main()
