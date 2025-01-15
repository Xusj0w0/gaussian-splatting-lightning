import os
import os.path as osp
import subprocess
from argparse import ArgumentParser
from dataclasses import asdict

import torch


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint file.")
    return parser


def main():
    args = make_parser().parse_args()
    args.ckpt_path = "outputs/vastgs-rubble-3_3-gsplat/1_1/checkpoints/epoch=73-step=30000.ckpt"
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    dataset_path = ckpt["datamodule_hyper_parameters"]["path"]
    dataparser_config_dict = asdict(ckpt["datamodule_hyper_parameters"]["parser"])
    output_path = osp.join(osp.dirname(ckpt["hyper_parameters"]["output_path"]), "evaluations")

    cmd = ["python", "main.py", "validate", "--save_val"]
    cmd += ["--ckpt_path", args.ckpt_path] + ["--verbose", "True"]
    cmd += (
        ["--data.path", dataset_path]
        + ["--data.parser", "Colmap"]
        + ["--data.parser.split_mode", "experiment"]
        + ["--data.parser.eval_image_select_mode", "list"]
        + ["--data.parser.eval_list", osp.join(dataset_path, "splits", "val_images.txt")]
        + ["--data.parser.scene_scale", str(dataparser_config_dict["scene_scale"])]
        + ["--data.parser.reorient", str(dataparser_config_dict["reorient"])]
        + ["--data.parser.appearance_groups", dataparser_config_dict["appearance_groups"] or "null"]
        + ["--data.parser.down_sample_factor", str(dataparser_config_dict["down_sample_factor"])]
        + ["--data.parser.down_sample_rounding_mode", dataparser_config_dict["down_sample_rounding_mode"]]
    )
    cmd += ["--output", output_path]
    subprocess.run(cmd)
    # + ["--data.parser.init_args.image_dir", dataparser_config_dict["image_dir"] or "null"]
    # + ["--data.parser.init_args.mask_dir", dataparser_config_dict["mask_dir"] or "null"]
    # + ["--data.parser.init_args.image_list", "null"]


if __name__ == "__main__":
    main()
