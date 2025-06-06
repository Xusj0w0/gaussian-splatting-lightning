import argparse
import os
import os.path as osp
import shutil
from typing import Dict, List, Tuple

from PIL import Image as PILImage

import internal.utils.colmap as colmap_utils
from internal.utils.colmap import Camera, Image


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="input directory")
    parser.add_argument("--output_dir", type=str)
    return parser


def modify_camera_sizes(args):
    sparse_model_path = osp.join(args.input_dir, "sparse")
    if not (osp.exists(osp.join(sparse_model_path, "images.bin")) or osp.exists(osp.join(sparse_model_path, "images.txt"))):
        sparse_model_path = osp.join(sparse_model_path, "0")
    if not (osp.exists(osp.join(sparse_model_path, "images.bin")) or osp.exists(osp.join(sparse_model_path, "images.txt"))):
        raise ValueError("Sparse model not found in {}".format(sparse_model_path))

    cameras: Dict[int, colmap_utils.Camera]; images: Dict[int, colmap_utils.Image]; points3D: Dict[int, colmap_utils.Point3D] # fmt: skip
    cameras, images, points3D = colmap_utils.read_model(sparse_model_path)

    for img_id, image in images.items():
        image_path = osp.join(args.input_dir, "images", image.name)
        assert osp.exists(image_path), "image not found"
        img = PILImage.open(image_path)
        width, height = img.width, img.height
        cam_id = image.camera_id
        camera = cameras[cam_id]
        if (width, height) != (camera.width, camera.height):
            cameras[cam_id] = Camera(id=camera.id, model=camera.model, width=width, height=height, params=camera.params)

    colmap_utils.write_model(cameras, images, points3D, args.output_dir)


def main():
    args = make_parser().parse_args()
    if args.output_dir is None or len(args.output_dir) == 0:
        args.output_dir = osp.join(args.input_dir, "modified")
    os.makedirs(args.output_dir, exist_ok=True)
    modify_camera_sizes(args)


if __name__ == "__main__":
    main()
