"""
for matrix city, set --down_sample_factor=1
for matrix city aerial and building, change
    --model.gaussian.init_args.optimization.means_lr_init
    --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final
    --model.gaussian.init_args.optimization.scales_lr
for example:
    python large_scene/CityGaussian/train_coarse.py \
        --dataset_path datasets/MegaNeRF/building/colmap \
        --project citygs-building \
        --down_sample_factor 4 \
        --model.gaussian.init_args.optimization.means_lr_init 0.00008 \
        --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 0.0000008 \
        --model.gaussian.init_args.optimization.scales_lr 0.0025
"""

import os.path as osp
import subprocess
from argparse import ArgumentParser


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--down_sample_factor", default=4)
    return parser


def main():
    args, unknown_args = make_parser().parse_known_args()
    subprocess.run(
        [
            "python",
            "main.py",
            "fit",
            "--project={}".format(args.project),
            "--output={}".format(f"outputs/{args.project}"),
            "-n={}".format("coarse"),
            "--data.parser=Colmap",
            "--data.path={}".format(args.dataset_path),
            "--data.parser.down_sample_factor={}".format(args.down_sample_factor),
            "--data.parser.image_list={}".format(osp.join(args.dataset_path, "splits/train_images.txt")),
            "--logger=tensorboard",
        ]
        + unknown_args
    )


if __name__ == "__main__":
    main()
