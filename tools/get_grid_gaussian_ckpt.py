import argparse
import os
import os.path as osp

import torch

from internal.cli import CLI
from internal.gaussian_splatting import GaussianSplatting
from myimpl.models.implicit_grid_gaussian import (ImplicitGridGaussian,
                                                  ImplicitLoDGridGaussian)
from myimpl.utils.grid_gaussian_loader import GridGaussianUtils


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path to orig scaffold/octree ckpt")
    parser.add_argument("--gspl_ckpt", type=str, default="tmp/implicit_lod_gaussian.ckpt")
    parser.add_argument("--output", type=str, required=True, help="output ckpt path")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    assert osp.exists(args.ckpt)
    assert osp.exists(args.gspl_ckpt), "gspl ckpt not found. can run `python main.py fit` to generate one"

    ckpt = torch.load(args.gspl_ckpt, map_location="cpu")

    modified_ckpt = GridGaussianUtils.convert_orig_to_ckpt(args.ckpt, ckpt)
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    torch.save(modified_ckpt, args.output)
