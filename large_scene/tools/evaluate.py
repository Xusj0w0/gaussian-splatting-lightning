# modified from utils/train_partitions.py

import argparse
import csv
import os
import os.path as osp
import time
from typing import Tuple

import torch
import yaml
from tqdm import tqdm

from internal.dataparsers import DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap
from internal.dataset import CacheDataLoader, Dataset
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from utils.common import AsyncImageSaver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--dataset_path", type=str, help="path to dataset")
    parser.add_argument("--down_sample_factor", type=int, default=4, help="down sample factor in training phase")
    parser.add_argument("--output", type=str, required=True, help="path to evaluation outputs")
    return parser.parse_args()


@torch.no_grad()
def validate(dataloader, gaussian_model, renderer, metric_calculator, image_saver):
    all_metrics = {}

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

            all_metrics[name], (predicted, gt) = metric_calculator(outputs, (camera, (name, image, mask), extra_data))

            image_saver(outputs, (camera, (name, image, mask), extra_data))

    return all_metrics


def get_metric_calculator(device, val_side: str = None):
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device=device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
    vgg_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device=device)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device=device)

    def get_metrics(predicts, batch):
        predicted_image = torch.clamp_max(predicts["render"], max=1.0)
        gt_image = batch[1][1]

        # mask
        if batch[1][-1] is not None:
            predicted_image = predicted_image * batch[1][-1]
            gt_image = gt_image * batch[1][-1]

        if val_side is not None:
            image_width = gt_image.shape[-1]
            half_width = image_width // 2

            if val_side == "left":
                width_from = 0
                width_to = half_width
            elif val_side == "right":
                width_from = half_width
                width_to = image_width
            else:
                raise RuntimeError()

            predicted_image = predicted_image[..., width_from:width_to]
            gt_image = gt_image[..., width_from:width_to]

        predicted_image = predicted_image.unsqueeze(0)
        gt_image = gt_image.unsqueeze(0)

        return {
            "psnr": psnr(predicted_image, gt_image),
            "ssim": ssim(predicted_image, gt_image),
            "vgg_lpips": vgg_lpips(predicted_image, gt_image),
            "alex_lpips": alex_lpips(predicted_image, gt_image),
        }, (predicted_image.squeeze(0), gt_image.squeeze(0))

    return get_metrics


class RendererWithMetrics(VanillaRenderer):
    def forward(self, viewpoint_camera, pc, bg_color, *args, **kwargs):
        start_t = time.time()
        outputs = super().forward(viewpoint_camera, pc, bg_color, *args, **kwargs)
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
        del ckpt

    elif args.ckpt.endswith(".ply"):
        assert args.dataset_path is not None, "ply model detected, dataset path should be specified"

        gaussian_model, _ = GaussianModelLoader.initialize_model_and_renderer_from_ply_file(
            args.ckpt, device, pre_activate=False
        )
        dataset_path = args.dataset_path
        dataparser_config = Colmap(
            split_mode="experiment",
            eval_image_select_mode="list",
            eval_list=osp.join(args.dataset_path, "splits/val_images.txt"),
            down_sample_factor=args.down_sample_factor,
        )

    renderer = RendererWithMetrics()
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
        render = (torch.clamp_max(predicts["render"], max=1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu()
        gt = (batch[1][1] * 255.0).to(torch.uint8).permute(1, 2, 0).cpu()
        montage = torch.cat([render, gt], dim=1)
        diff = torch.abs(render.float() - gt.float())

        for d in ["render", "gt", "montage", "diff"]:
            os.makedirs(os.path.join(output_dir, d), exist_ok=True)

        async_image_saver.save(render.numpy(), os.path.join(output_dir, "render", "{}.png".format(batch[1][0])))
        async_image_saver.save(gt.numpy(), os.path.join(output_dir, "gt", "{}.png".format(batch[1][0])))
        async_image_saver.save(montage.numpy(), os.path.join(output_dir, "montage", "{}.png".format(batch[1][0])))
        async_image_saver.save(diff.numpy(), os.path.join(output_dir, "diff", "{}.png".format(batch[1][0])))

    try:
        metrics = validate(dataloader, gaussian_model, renderer, get_metric_calculator(device), image_saver)
    finally:
        async_image_saver.stop()

    print("Repeat rendering for evaluating FPS...")
    cameras = [camera for camera, _, _ in dataloader]
    bg_color = torch.zeros((3,), dtype=torch.float, device=cameras[0].device)
    n_gaussian_list = []
    time_list = []
    n_rendered_frames = 0
    for _ in range(8):
        for camera in cameras:
            predicts = renderer(
                camera,
                gaussian_model,
                bg_color,
            )
            n_gaussian_list.append(predicts["n_gaussians"])
            time_list.append(predicts["time"])
            n_rendered_frames += 1

    metric_list_key_by_name = {}
    available_metric_keys = ["psnr", "ssim", "vgg_lpips", "alex_lpips"]
    with open(os.path.join(output_dir, "metrics-{}.csv".format(os.path.basename(output_dir))), "w") as f:
        metrics_writer = csv.writer(f)
        metrics_writer.writerow(["name"] + available_metric_keys)
        for image_name, image_metrics in metrics.items():
            metric_row = [image_name]
            for k in available_metric_keys:
                v = image_metrics[k]
                metric_list_key_by_name.setdefault(k, []).append(v)
                metric_row.append(v.item())

            metrics_writer.writerow(metric_row)

        metrics_writer.writerow([""] * len(available_metric_keys))

        mean_row = ["MEAN"]
        for k in available_metric_keys:
            mean_row.append(torch.mean(torch.stack(metric_list_key_by_name[k]).float()).item())
        metrics_writer.writerow(mean_row)

        average_n_gaussians = torch.mean(torch.tensor(n_gaussian_list, dtype=torch.float)).item()
        fps = n_rendered_frames / torch.sum(torch.tensor(time_list, dtype=torch.float))
        metrics_writer.writerow(["FPS", "{}".format(fps)])
        metrics_writer.writerow(["AverageNGaussians", "{}".format(average_n_gaussians)])

        print(mean_row)
        print("FPS={}, AverageNGaussians={}".format(fps, average_n_gaussians))


main()
