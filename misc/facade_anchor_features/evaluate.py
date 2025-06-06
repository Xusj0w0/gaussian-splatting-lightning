# modified from utils/train_partitions.py

import argparse
import csv
import os
import os.path as osp
import time
from typing import Tuple

import cv2
import torch
import torch.nn.functional as F
import yaml
from gsplat.rasterize_to_weights import rasterize_to_weights
from torch_scatter import scatter_sum
from tqdm import tqdm

from internal.dataparsers import DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap
from internal.dataset import CacheDataLoader, Dataset
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers import Renderer
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.visualizers import Visualizers
from myimpl.renderers.grid_renderer import (GridGaussianRenderer,
                                            GridGaussianRendererModule,
                                            GridRendererUtils)
from utils.common import AsyncImageSaver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint file",
        default="misc/facade_anchor_features/outputs/checkpoints/epoch=24-step=60000.ckpt",
    )
    parser.add_argument("--dataset_path", type=str, help="path to dataset", default="datasets/MegaNeRF/residence")
    parser.add_argument("--down_sample_factor", type=int, default=4, help="down sample factor in training phase")
    parser.add_argument(
        "--output", type=str, help="path to evaluation outputs", default="misc/facade_anchor_features/evaluations"
    )
    return parser.parse_args()


class BlendRenderer(GridGaussianRenderer):
    def instantiate(self, *args, **kwargs):
        return BlendRendererModule(self)


class BlendRendererModule(GridGaussianRendererModule):
    def render_feature(self, properties_list, viewpoint_camera, pc, scaling_modifier=1, **kwargs):
        projections_list, isects_list, visibility_filter, preprocessed_camera = (
            GridRendererUtils.project_to_pixels_loop(
                self,
                properties_list,
                viewpoint_camera,
                scaling_modifier=scaling_modifier,
                return_preprocessed_cam=True,
                **kwargs,
            )
        )
        projections = GridRendererUtils.concatenate_projections(projections_list, isects_list)
        means2d, *_ = projections

        input_features = means2d.new_zeros((0, pc.config.feature_dim))
        input_opacities = means2d.new_zeros((0,))
        for cam_id in range(len(viewpoint_camera)):
            _, _, _, _, _opacities, _anchor_mask, _primitive_mask, *_ = properties_list[cam_id]
            _visibility_filter = visibility_filter[cam_id]
            indices = torch.nonzero(_anchor_mask, as_tuple=True)[0]
            indices = indices.reshape(-1, 1).expand(-1, pc.n_offsets).reshape(-1)
            indices = indices[_primitive_mask]
            indices = indices[_visibility_filter]
            features = pc.get_anchor_features[indices]

            input_opacities = torch.cat([input_opacities, _opacities[_visibility_filter]], dim=0)
            input_features = torch.cat([input_features, features], dim=0)

        render_feature, alpha = GridRendererUtils.rasterize_cat_projections(
            preprocessed_camera=preprocessed_camera,
            projections=projections,
            properties=(input_features, input_opacities),
            bg_color=means2d.new_zeros((len(viewpoint_camera), pc.config.feature_dim)),
            tile_size=self.config.block_size,
        )
        aligned_feature = None
        feature_adapter = getattr(pc, "get_feature_adapter_mlp", None)
        if feature_adapter is not None:
            aligned_feature = feature_adapter(render_feature)

        left_weights, right_weights = self.rasterize_cat_projections_blend(
            preprocessed_camera=preprocessed_camera,
            projections=projections,
            properties=(input_features, input_opacities),
            bg_color=means2d.new_zeros((len(viewpoint_camera), pc.config.feature_dim)),
            tile_size=self.config.block_size,
        )
        if left_weights is not None and right_weights is not None:
            indices = (
                torch.nonzero(_anchor_mask, as_tuple=False)
                .squeeze()
                .reshape(-1, 1)
                .expand(-1, pc.n_offsets)
                .reshape(-1)
            )
            indices = indices[_primitive_mask][_visibility_filter]
            # self.left_primitive_weights = scatter_sum(left_weights, indices, dim=0, dim_size=pc.get_anchors.shape[0])
            # self.right_primitive_weights = scatter_sum(right_weights, indices, dim=0, dim_size=pc.get_anchors.shape[0])
            self.left_primitive_weights = left_weights
            self.right_primitive_weights = right_weights
            self.indices = indices

        return render_feature, aligned_feature, alpha

    @torch.no_grad()
    def rasterize_cat_projections_blend(
        self,
        preprocessed_camera: tuple,
        projections: tuple,
        properties: tuple,
        bg_color: torch.Tensor,
        tile_size: int = 16,
    ):
        if left_mask is not None and right_mask is not None:
            _, _, (image_width, image_height) = preprocessed_camera
            means2d, conics, isects_offsets, flatten_ids = projections
            colors, opacities = properties

            left_weights, _, _, _ = rasterize_to_weights(
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                image_width=image_width[0],
                image_height=image_height[0],
                tile_size=tile_size,
                isect_offsets=isects_offsets,
                flatten_ids=flatten_ids,
                pixel_weights=F.interpolate(left_mask, (image_height[0], image_width[0])).squeeze(),
                packed=True,
            )
            right_weights, _, _, _ = rasterize_to_weights(
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                image_width=image_width[0],
                image_height=image_height[0],
                tile_size=tile_size,
                isect_offsets=isects_offsets,
                flatten_ids=flatten_ids,
                pixel_weights=F.interpolate(right_mask, (image_height[0], image_width[0])).squeeze(),
                packed=True,
            )

            return left_weights, right_weights
        return None, None


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

    renderer_params = ckpt["hyper_parameters"]["renderer"]
    renderer = BlendRenderer(**{k: getattr(renderer_params, k) for k in renderer_params.__dataclass_fields__})
    renderer.render_feature_size = 907
    renderer = renderer.instantiate()
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

    bg_color = torch.zeros((3,), dtype=torch.float, device="cuda")
    with tqdm(dataloader) as t:
        for camera, (name, image, mask), extra_data in t:
            if name != "000099.JPG":
                continue
            image = image.to(device="cuda")
            if mask is not None:
                mask = mask.to(device="cuda")
            outputs = renderer(
                camera,
                gaussian_model,
                bg_color,
                render_types=["rgb", "feature"],
            )

        anchor_features = gaussian_model.get_anchor_features
        left_weights = renderer._renderer.left_primitive_weights
        right_weights = renderer._renderer.right_primitive_weights
        # features_left = anchor_features[left_weights > 100]
        # features_right = anchor_features[right_weights > 100]
        features = anchor_features[renderer._renderer.indices]
        features_left = features[left_weights > 100]
        features_right = features[right_weights > 100]
        torch.save({"left": features_left, "right": features_right}, "misc/facade_anchor_features/facade_features.pt")

        # anchor_weights = renderer._renderer.anchor_weights
        # anchors = gaussian_model.get_anchors
        # anchors_homo = torch.cat([anchors, anchors.new_ones((anchors.shape[0], 1))], -1)
        # anchors_ndc = anchors_homo @ camera.full_projection
        # anchors_2d = anchors_ndc[:, :2] / anchors_ndc[:, 3:]
        # anchors_pix = (anchors_2d + 1) / 2

        # in_left_mask = F.grid_sample(left_mask, anchors_pix[None, None], align_corners=True).squeeze() > 0
        # in_right_mask = F.grid_sample(right_mask, anchors_pix[None, None], align_corners=True).squeeze() > 0
        # in_left_facade = in_left_mask & (anchor_weights > 0.01)
        # in_right_facade = in_right_mask & (anchor_weights > 0.01)


device = torch.device("cuda")
left_mask = cv2.imread("misc/facade_anchor_features/left-facade-mask.png")
left_mask = (torch.from_numpy(left_mask).to(device) > 0).any(dim=-1)[None, None, ...].float()
right_mask = cv2.imread("misc/facade_anchor_features/right-facade-mask.png")
right_mask = (torch.from_numpy(right_mask).to(device) > 0).any(dim=-1)[None, None, ...].float()

main()
