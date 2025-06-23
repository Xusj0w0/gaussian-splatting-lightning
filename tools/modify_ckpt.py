import argparse
import os
import os.path as osp

import numpy as np
import torch

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.partitioning_utils import MinMaxBoundingBox


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", type=str, required=True)
    parser.add_argument("--ckpt_out", type=str)
    return parser


def load_from_ckpt(ckpt, device):
    return GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device)


def main():
    args = make_parser().parse_args()
    if args.ckpt_out is None:
        basename = osp.basename(args.ckpt)
        name, ext = basename.rsplit(".", 1)
        args.ckpt_out = osp.join(osp.dirname(args.ckpt), f"{name+'-modified'}.{ext}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt["hyper_parameters"]["renderer"].filter_by_vs_size = True
    ckpt["hyper_parameters"]["renderer"].filter_by_ws_size = True
    torch.save(ckpt, args.ckpt_out)


if __name__ == "__main__":
    main()
