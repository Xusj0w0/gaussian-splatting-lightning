import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from external.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from utils.common import (AsyncImageReader, AsyncImageSaver, AsyncNDArraySaver,
                          find_files)
from utils.distibuted_tasks import (configure_arg_parser,
                                    get_task_list_with_args)


def make_parser():
    parser = argparse.ArgumentParser(description="Get HxWxC features. min(H, W) == 518 / 14")
    parser.add_argument("image_dir")
    parser.add_argument("--input_size", "-s", type=int, default=518)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--encoder", default="vitl")
    parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
    parser.add_argument("--da2_ckpt", "-c", type=str, default="checkpoints/da2/depth_anything_v2_vitl.pth")
    configure_arg_parser(parser)
    return parser


def build_da2_backbone(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(args.da2_ckpt, map_location="cpu"))
    depth_anything = depth_anything.to(DEVICE).eval()

    return depth_anything


if __name__ == "__main__":
    args = make_parser().parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.image_dir), "features_from_depth")

    images = get_task_list_with_args(args, find_files(args.image_dir, args.extensions, as_relative_path=False))
    assert len(images) > 0, "not an image with extension name '{}' can be found in '{}'".format(
        args.extensions, args.image_dir
    )

    depth_anything = build_da2_backbone(args)

    ndarray_saver = AsyncNDArraySaver()
    image_reader = AsyncImageReader(image_list=images)

    class FeatureSaver:
        def __init__(self, ndarray_saver):
            self.output_filename = None
            self.ndarray_saver = ndarray_saver

        def __call__(self, module, input, output):
            self.ndarray_saver.save(input[0].clone().detach().cpu().numpy(), self.output_filename)

    hook = FeatureSaver(ndarray_saver)
    handle = depth_anything.depth_head.scratch.output_conv1.register_forward_hook(hook)
    try:
        with torch.no_grad(), tqdm(range(len(images))) as t:
            for _ in t:
                image_path, raw_image = image_reader.get()
                image_name = image_path[len(args.image_dir) :].lstrip(os.path.sep)
                output_filename = os.path.join(args.output, "{}.npy".format(image_name))

                hook.output_filename = output_filename

                depth_anything.infer_image(raw_image, args.input_size)

    finally:
        ndarray_saver.stop()
        image_reader.stop()
        handle.remove()
