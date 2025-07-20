import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import cv2
import numpy as np

from internal.utils.colmap import (CAMERA_MODEL_NAMES, CAMERA_MODELS, Camera,
                                   Image, Point3D, read_model, write_model)


def make_parser():
    parser = ArgumentParser(description="Undistort images using COLMAP camera parameters")
    parser.add_argument(
        "--sparse_model_path", type=str, required=True, help="Path to the distorted COLMAP model directory"
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing distorted images")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save undistorted sparse model and images"
    )
    return parser


def get_intrinsic_matrix(camera: Camera):
    if camera.model == "SIMPLE_RADIAL":
        fx = fy = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
        k1 = camera.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        D = np.array([k1, 0, 0, 0])

    elif camera.model == "PINHOLE":
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        D = np.zeros(4)

    elif camera.model == "OPENCV":
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
        k1 = camera.params[4]
        k2 = camera.params[5]
        p1 = camera.params[6]
        p2 = camera.params[7]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        D = np.array([k1, k2, p1, p2])

    else:
        raise NotImplementedError(f"Camera model {camera.model} not supported.")

    return K, D


def undistort_images(sparse_model_path: str, image_dir: str, output_dir: str):
    output_sparse_path = os.path.join(output_dir, "sparse")
    output_image_dir = os.path.join(output_dir, "images")
    os.makedirs(output_sparse_path, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    points3d: Dict[int, Point3D]
    cameras, images, points3d = read_model(sparse_model_path)

    intrinsic_dict = {}
    for camera_id, camera in cameras.items():
        K, D = get_intrinsic_matrix(camera)
        intrinsic_dict[camera_id] = (K, D)

    # save undistorted images
    for image_id, image in images.items():
        image_path = os.path.join(image_dir, image.name)
        if not os.path.exists(image_path):
            print(f"Image {image.name} does not exist in {image_dir}. Skipping.")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image {image.name}. Skipping.")
            continue

        K, D = intrinsic_dict[image.camera_id]
        h, w = img.shape[:2]

        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
        undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        undistorted_image_path = os.path.join(output_image_dir, image.name)
        cv2.imwrite(undistorted_image_path, undistorted_img)

    # convert sparse model to undistorted one
    new_cameras = {}
    for cam_id, cam in cameras:
        K, D = intrinsic_dict[cam_id]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        new_cameras[cam_id] = Camera(
            model=CAMERA_MODEL_NAMES["PINHOLE"], width=cam.width, height=cam.height, params=np.array([fx, fy, cx, cy])
        )

    new_images = {}
    for image_id, image in images.items():
        K, D = intrinsic_dict[image.camera_id]
        if len(image.xys) > 0:
            xys_undistorted = cv2.undistortPoints(image.xys.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        else:
            xys_undistorted = image.xys  # empty
        new_images[image_id] = Image(
            id=image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,  # same ID, since we just replaced its definition
            name=image.name,
            xys=xys_undistorted,
            point3D_ids=image.point3D_ids
        )
    
    new_points3d =points3d
    write_model(new_cameras, new_images, new_points3d, output_sparse_path)

def main():
    args = make_parser().parse_args()
    undistort_images(args.sparse_model_path, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()