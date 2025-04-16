# modified from utils/train_partitions.py

import argparse
import csv
import os
import os.path as osp
import time
from dataclasses import asdict
from typing import Tuple

import torch
import yaml
from tqdm import tqdm

from internal.dataparsers import DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap
from internal.dataset import CacheDataLoader, Dataset
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers import Renderer
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.visualizers import Visualizers
from utils.common import AsyncImageSaver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--dataset_path", type=str, help="path to dataset")
    parser.add_argument("--down_sample_factor", type=int, default=4, help="down sample factor in training phase")
    parser.add_argument("--output", type=str, required=True, help="path to evaluation outputs")
    return parser.parse_args()


@torch.no_grad()
def render_feature(dataloader, gaussian_model, renderer, image_saver):
    bg_color = torch.zeros((3,), dtype=torch.float, device="cuda")
    with tqdm(dataloader) as t:
        for camera, (name, image, mask), extra_data in t:
            image = image.to(device="cuda")
            if mask is not None:
                mask = mask.to(device="cuda")
            outputs = renderer(
                camera,
                gaussian_model,
                bg_color,
            )
            image_saver(outputs, (camera, (name, image, mask), extra_data))


class RendererWithMetricsWrapper(Renderer):
    def __init__(self, renderer):
        super().__init__()
        self._renderer = renderer

    def forward(self, viewpoint_camera, pc, bg_color, *args, **kwargs):
        start_t = time.time()
        outputs = self._renderer(viewpoint_camera, pc, bg_color, *args, **kwargs)
        outputs["time"] = time.time() - start_t
        outputs["n_gaussians"] = pc.n_gaussians
        return outputs


def load_from_ckpt(args, device) -> Tuple[VanillaGaussianModel, VanillaRenderer, DataParserOutputs]:
    if args.ckpt.endswith(".ckpt"):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device)
        dataset_path = ckpt["datamodule_hyper_parameters"]["path"]
        dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
        dataparser_config.image_list = None
        if args.dataset_path is not None:
            dataset_path = args.dataset_path
            dataparser_config.eval_list = osp.join(args.dataset_path, "splits/val_images.txt")

    elif args.ckpt.endswith(".ply"):
        raise NotImplementedError("Loading from .ply is not supported yet.")

    from myimpl.renderers.grid_feature_renderer import \
        GridFeatureGaussianRenderer

    ckpt_renderer = ckpt["hyper_parameters"]["renderer"]
    params = {
        k: getattr(ckpt_renderer, k)
        for k in GridFeatureGaussianRenderer.__dataclass_fields__
        if k in ckpt_renderer.__dict__
    }
    renderer = GridFeatureGaussianRenderer(**params).instantiate()
    renderer.setup(stage="validation")
    renderer = RendererWithMetricsWrapper(renderer)
    # avoid loading point cloud
    dataparser_config.points_from == "random"
    dataparser_outputs: DataParserOutputs = dataparser_config.instantiate(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()
    return gaussian_model, renderer, dataparser_outputs


def main():
    args = parse_args()
    device = torch.device("cuda")
    gaussian_model, renderer, dataparser_outputs = load_from_ckpt(args, device)

    # setup DataLoader
    dataloader = CacheDataLoader(
        Dataset(
            dataparser_outputs.test_set,
            undistort_image=False,
            camera_device=torch.device("cuda"),
            image_device=torch.device("cpu"),
            allow_mask_interpolation=True,
        ),
        max_cache_num=64,
        shuffle=False,
        num_workers=2,
    )
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # image saver
    async_image_saver = AsyncImageSaver(is_rgb=True)

    def image_saver(predicts, batch):
        gt = (batch[1][1] * 255.0).to(torch.uint8).permute(1, 2, 0).cpu()
        feature = Visualizers.pca_colormap(predicts["render_feature"].permute(2, 0, 1)) * 255.0  # clamped
        feature_resized = torch.nn.functional.interpolate(
            feature.unsqueeze(0), gt.shape[:2], mode="bilinear", align_corners=True
        ).squeeze(0)
        feature = feature.permute(1, 2, 0).to(torch.uint8).cpu()
        feature_resized = feature_resized.permute(1, 2, 0).to(torch.uint8).cpu()
        montage = torch.cat([feature_resized, gt], dim=1)

        for d in ["montage_feature", "feature"]:
            os.makedirs(os.path.join(output_dir, d), exist_ok=True)

        async_image_saver.save(
            montage.numpy(), os.path.join(output_dir, "montage_feature", "{}.png".format(batch[1][0]))
        )
        async_image_saver.save(feature.numpy(), os.path.join(output_dir, "feature", "{}.png".format(batch[1][0])))

    try:
        render_feature(dataloader, gaussian_model, renderer, image_saver)
    finally:
        async_image_saver.stop()


main()
