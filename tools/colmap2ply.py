import argparse
import os
import os.path as osp
from typing import Dict

import numpy as np
from plyfile import PlyData, PlyElement

import internal.utils.colmap as colmap_utils


def make_parser():
    parser = argparse.ArgumentParser(description="Convert COLMAP point loud to PLY format")
    parser.add_argument("input_path", type=str, help="Path to the input COLMAP sparse model")
    parser.add_argument("--output_path", type=str, help="Path to the output PLY file", default=None)
    return parser


def save_plyfile(filename, points3D: Dict):
    xyz = np.stack([v.xyz for v in points3D.values()], axis=0)
    rgb = np.stack([v.rgb for v in points3D.values()], axis=0)

    dtype_full = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(len(points3D), dtype=dtype_full)
    attributes = np.concatenate([xyz, rgb], axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=True).write(filename)


if __name__ == "__main__":
    args = make_parser().parse_args()

    # detect model type
    model_type = ".bin"
    if not osp.exists(osp.join(args.input_path, "images.bin")):
        if not osp.exists(osp.join(args.input_path, "images.txt")):
            raise ValueError("Cannot detect model type")
        model_type = ".txt"

    if args.output_path is None:
        args.output_path = osp.join(args.input_path, "point_cloud.ply")
    if model_type == ".bin":
        points3D = colmap_utils.read_points3D_binary(osp.join(args.input_path, "points3D.bin"))
    else:
        points3D = colmap_utils.read_points3D_text(osp.join(args.input_path, "points3D.txt"))

    save_plyfile(args.output_path, points3D)
