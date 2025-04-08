import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from external.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from internal.utils.seganygs import SegAnyGSUtils
from internal.utils.visualizers import Visualizers
from utils.common import (AsyncImageReader, AsyncImageSaver, AsyncNDArraySaver,
                          AsyncTensorSaver, find_files)
from utils.distibuted_tasks import (configure_arg_parser,
                                    get_task_list_with_args)


def make_parser():
    parser = argparse.ArgumentParser(description="Get HWC features. min(H, W) == 518 / 14")
    parser.add_argument("image_dir")
    parser.add_argument("--input_size", "-s", type=int, default=518)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--encoder", default="vitl")
    parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
    parser.add_argument("--da2_ckpt", "-c", type=str, default="checkpoints/da2/depth_anything_v2_vitl.pth")
    parser.add_argument("--preview", action="store_true", default=False)
    parser.add_argument("--colormap", type=str, default="default")
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


class FeatureGetter:
    def __init__(self, da_model: DepthAnythingV2):
        self._handle = da_model.depth_head.scratch.output_conv1.register_forward_hook(self)
        self._embedding = None

    def __call__(self, module, input, output):
        self._embedding = input[0].squeeze()

    def get_image_embedding(self):
        return self._embedding

    def __del__(self):
        self._handle.remove()
        super().__del__()


def apply_color_map(normalized_depth, colormap):
    colored_depth = Visualizers.float_colormap(torch.from_numpy(normalized_depth).unsqueeze(0), colormap=colormap)
    colored_depth = (colored_depth.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return colored_depth


if __name__ == "__main__":
    args = make_parser().parse_args()

    image_path = args.image_dir
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path.rstrip("/")))
    print(f"output_path={os.path.join(output_path, 'semantic')}")
    depth_dir = os.path.join(output_path, "semantic", "depth")
    depth_preview_dir = os.path.join(output_path, "semantic", "depth_preview")
    feature_dir = os.path.join(output_path, "semantic", "depth_feature")
    feature_preview_dir = os.path.join(output_path, "semantic", "depth_feature_preview")

    for d in [depth_dir, feature_dir]:
        os.makedirs(d, exist_ok=True)
    if args.preview:
        for d in [depth_preview_dir, feature_preview_dir]:
            os.makedirs(d, exist_ok=True)

    images = get_task_list_with_args(args, find_files(args.image_dir, args.extensions, as_relative_path=False))
    assert len(images) > 0, "not an image with extension name '{}' can be found in '{}'".format(
        args.extensions, args.image_dir
    )

    depth_anything = build_da2_backbone(args)
    feature_getter = FeatureGetter(da_model=depth_anything)

    image_reader = AsyncImageReader(image_list=images)
    ndarray_saver = AsyncNDArraySaver()
    image_saver = AsyncImageSaver(is_rgb=True)

    try:
        with torch.no_grad(), tqdm(range(len(images))) as t:
            for _ in t:
                image_path, raw_image = image_reader.get()
                image_name = image_path[len(args.image_dir) :].lstrip(os.path.sep)
                output_filename = os.path.join(args.output, "{}.npy".format(image_name))

                depth = depth_anything.infer_image(raw_image, args.input_size)  # [H, W]
                normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
                image_embedding = feature_getter.get_image_embedding()

                ndarray_saver.save(normalized_depth, os.path.join(depth_dir, "{}.npy".format(image_name)))
                image_embedding_np = image_embedding.permute(1, 2, 0).clone().detach().cpu().numpy()
                ndarray_saver.save(image_embedding_np, os.path.join(feature_dir, "{}.npy".format(image_name)))

                if args.preview:
                    image_saver.save(
                        normalized_depth,
                        os.path.join(depth_preview_dir, "{}.png".format(image_name)),
                        processor=lambda x: apply_color_map(x, colormap=args.colormap),
                    )

                    image_embedding_flatten_normalized = torch.nn.functional.normalize(
                        image_embedding.permute(1, 2, 0).reshape((-1, image_embedding.shape[0])), dim=-1
                    )
                    pca_projection_matrix = SegAnyGSUtils.get_pca_projection_matrix(
                        semantic_features=image_embedding_flatten_normalized
                    )
                    pca_color = SegAnyGSUtils.get_pca_projected_colors(
                        image_embedding_flatten_normalized, pca_projection_matrix
                    )
                    feature_preview = pca_color.reshape((image_embedding.shape[1], image_embedding.shape[2], -1))
                    feature_preview = (feature_preview * 255).to(torch.uint8).cpu().numpy()
                    image_saver.save(feature_preview, os.path.join(feature_preview_dir, f"{image_name}.png"))

    finally:
        ndarray_saver.stop()
        image_reader.stop()
