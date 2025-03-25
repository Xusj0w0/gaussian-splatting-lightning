import argparse
import json
import os
import os.path as osp
from typing import Dict, List

import numpy as np

import internal.utils.colmap as colmap_utils


def make_parser():
    parser = argparse.ArgumentParser(description="Write camera json file from colmap")
    parser.add_argument("--path", type=str, required=True, help="path to colmap folder")
    parser.add_argument("--image_list_path", type=str, default=None, help="path to image list file")
    parser.add_argument("--output_path", type=str, required=True, help="path to output json file")
    return parser


def detect_sparse_model_dir(args) -> str:
    if os.path.isdir(os.path.join(args.path, "sparse", "0")):
        return os.path.join(args.path, "sparse", "0")
    return os.path.join(args.path, "sparse")


def read_cameras(args):
    sparse_model_dir = detect_sparse_model_dir(args)
    if osp.exists(osp.join(sparse_model_dir, "cameras.bin")):
        cameras = colmap_utils.read_cameras_binary(osp.join(sparse_model_dir, "cameras.bin"))
        images = colmap_utils.read_images_binary(osp.join(sparse_model_dir, "images.bin"))
    elif osp.exists(osp.join(sparse_model_dir, "cameras.txt")):
        cameras = colmap_utils.read_cameras_text(osp.join(sparse_model_dir, "cameras.txt"))
        images = colmap_utils.read_images_text(osp.join(sparse_model_dir, "images.txt"))
    else:
        raise ValueError("Cameras file not found in sparse model directory")
    return cameras, images


def get_intrinsics(params, model):
    if model == "SIMPLE_PINHOLE":
        return {"fx": params[0], "fy": params[0], "cx": params[1], "cy": params[2]}
    elif model == "PINHOLE":
        return {"fx": params[0], "fy": params[1], "cx": params[2], "cy": params[3]}
    else:
        raise RuntimeError


def write_cameras(args, cameras: Dict[int, colmap_utils.Camera], images: Dict[int, colmap_utils.Image]):
    camera_list = []
    if args.image_list_path is not None:
        with open(args.image_list_path, "r", encoding="utf-8") as f:
            image_list = [line.strip() for line in f.readlines()]
    else:
        image_list = []

    for image_id, image in images.items():
        if image_list is not None and len(image_list) > 0:
            if image.name not in image_list:
                continue

        camera = cameras[image.camera_id]
        R = image.qvec2rotmat()
        T = np.array(image.tvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)
        cam_info = {
            "id": image.id,
            "img_name": image.name,
            "width": int(camera.width),
            "height": int(camera.height),
            "position": c2w[:3, -1].tolist(),
            "rotation": c2w[:3, :3].tolist(),
            "time": None,
            "appearance_id": None,
            "normalized_appearance_id": None,
        }
        cam_info.update(get_intrinsics(camera.params, camera.model))
        camera_list.append(cam_info)

    with open(osp.join(args.output_path, "cameras.json"), "w") as f:
        json.dump(camera_list, f, indent=4, ensure_ascii=False)


def main(args):
    cameras, images = read_cameras(args)
    write_cameras(args, cameras, images)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
