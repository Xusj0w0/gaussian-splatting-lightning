import sys

sys.path.insert(0, "external/dinov2")
import argparse
import importlib
import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from external.dinov2.dinov2.models.vision_transformer import \
    DinoVisionTransformer
from internal.utils.seganygs import SegAnyGSUtils
from utils.common import AsyncImageReader, AsyncImageSaver, AsyncNDArraySaver
from utils.distibuted_tasks import (configure_arg_parser,
                                    get_task_list_with_args)


def make_parser():
    parser = argparse.ArgumentParser(description="Extract HWC feature to npy file")
    parser.add_argument("image_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dino2_arch", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--preview", action="store_true", default=False)
    parser.add_argument("--ext", "-e", nargs="+", default=["jpg", "jpeg", "JPG", "JPEG"])
    configure_arg_parser(parser)
    return parser


def build_dinov2(args, device="cpu") -> DinoVisionTransformer:
    # dino = torch.hub.load("facebookresearch/dinov2", args.dino2_arch)
    try:
        import external.dinov2.hubconf as dino_hubconf

        entry = dino_hubconf.__dict__[args.dino2_arch]
        dino: DinoVisionTransformer = entry().to(device).eval()
    except:
        print("Failed to load DINOv2 from external.dinov2.hubconf, using default")
        pass
    return dino


def transform_image(img: np.ndarray, img_size: int = 224) -> torch.Tensor:
    img = Image.fromarray(img)
    img = TF.resize(img, (img_size, img_size), interpolation=TF.InterpolationMode.BILINEAR)
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0)
    return img


def get_image_list(args):
    image_name_set = {}
    for e in args.ext:
        for i in glob(os.path.join(image_path, f"**/*.{e}"), recursive=True):
            image_name_set[i] = True
    image_list = list(image_name_set.keys())
    assert len(image_list) > 0, "Not a image can be found"
    print(f"{len(image_list)} images found")
    image_list.sort()

    image_list = get_task_list_with_args(args, image_list)
    return image_list


if __name__ == "__main__":
    args = make_parser().parse_args()

    MODEL_DEVICE = "cuda"

    # build paths
    image_path = args.image_path
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path.rstrip("/")))
    print(f"output_path={os.path.join(output_path, 'semantic')}")

    # build output dirs
    feature_dir = os.path.join(output_path, "semantic", "dinov2")
    feature_preview_dir = os.path.join(output_path, "semantic", "dinov2_preview")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(feature_preview_dir, exist_ok=True)

    # initialize SAM
    print("Initializing DINOv2...")
    predictor = build_dinov2(args, device=MODEL_DEVICE)

    print("Finding image files...")
    image_list = get_image_list(args)

    image_reader = AsyncImageReader(image_list)
    image_saver = AsyncImageSaver()
    ndarray_saver = AsyncNDArraySaver()

    try:
        with tqdm(range(len(image_list))) as t:
            for _ in t:
                # get image information
                image_full_path, img = image_reader.queue.get()
                image_name = image_full_path[len(image_path) :].lstrip("/")

                t.set_description(f"{image_name}")
                semantic_file_name = f"{image_name}.npy"

                # predict image embedding
                img_tensor = transform_image(img, 448).to(MODEL_DEVICE)
                feature_size = (torch.tensor(img_tensor.shape[-2:]) / predictor.patch_size).int().tolist()
                with torch.no_grad():
                    image_embedding = (
                        predictor.forward_features(img_tensor)["x_norm_patchtokens"].squeeze(0).permute(1, 0)
                    )
                image_embedding = image_embedding.reshape(-1, *feature_size)

                # save
                image_embedding_np = image_embedding.permute(1, 2, 0).clone().detach().cpu().numpy()
                ndarray_saver.save(image_embedding_np, os.path.join(feature_dir, semantic_file_name))

                # preview
                if args.preview is True:
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
        image_reader.stop()
        image_saver.stop()
        ndarray_saver.stop()
