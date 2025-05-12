from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from gsplat.utils import depth_to_points
from torchmetrics.image import StructuralSimilarityIndexMeasure

from internal.cameras.cameras import Camera
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from myimpl.utils.cameras import InstantiatedCameras
from myimpl.utils.dataset_utils import (DepthData, ExtraDataProcessorOutputs,
                                        MaskData, SemanticData)
from myimpl.utils.loss_utils import (DepthRegularization, MultiView,
                                     WeightScheduler)

__all__ = ["ScaffoldMetrics", "ScaffoldMetricsImpl"]


@dataclass
class ScaffoldMetrics(VanillaMetrics):
    lambda_dreg: float = 0.01

    lambda_flatten: float = 100.0

    lambda_normal: float = 0.015

    normal_from_iter: int = 7_000

    grad_weighted_normal: bool = False

    multiview_from_iter: int = 7_000

    lambda_multiview: WeightScheduler = field(
        default_factory=lambda: {
            "init": 0.01,
            "final_factor": 1.0,
            "mode": "linear",
        }
    )

    lambda_pixshift: float = 0.03

    # multiview_loss_func: PatchMultiviewLoss = field(default_factory=lambda: PatchMultiviewLoss())

    # feature regularization
    lambda_feature: WeightScheduler = field(
        default_factory=lambda: {
            "init": 0.5,
            "final_factor": 0.01,
            "mode": "linear",
        }
    )

    # depth regularization
    lambda_depth: WeightScheduler = field(
        default_factory=lambda: {
            "init": 1.0,
            "final_factor": 0.01,
            "mode": "exp",
        }
    )

    depth_loss_func: DepthRegularization = field(default_factory=lambda: DepthRegularization())

    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs):
        return ScaffoldMetricsImpl(self)


class ScaffoldMetricsImpl(VanillaMetricsImpl):
    config: ScaffoldMetrics

    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)

        for k in self.config.__dataclass_fields__:
            v = getattr(self.config, k)
            if isinstance(v, WeightScheduler) and v.max_steps is None:
                v.max_steps = pl_module.trainer.max_steps
        self.render_depth = "acc_depth" in pl_module.renderer_output_types

    @staticmethod
    def _create_fused_ssim_adapter():
        # fmt: off
        from fused_ssim import fused_ssim
        def adapter(pred, gt):
            if len(pred.shape) == 3:
                pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
            return fused_ssim(pred, gt)
        # fmt: on
        return adapter

    def _get_basic_metrics(
        self,
        pl_module,
        gaussian_model,
        batch: Tuple[Camera, Tuple[str, torch.Tensor, Optional[torch.Tensor]], ExtraDataProcessorOutputs],
        outputs,
    ):
        global_step = pl_module.trainer.global_step + 1
        _, image_info, extra_data = batch
        image_name, gt_image, mask = image_info

        # if single batch
        if isinstance(image_name, str):
            gt_image = gt_image.unsqueeze(0)
            extra_data = {k: [v] for k, v in extra_data.items()}

        # basic metrics
        render = outputs["render"]
        render_aug = outputs.get("render_aug", None)
        if mask is not None:
            _mask = mask.to(torch.uint8)
            gt_image = gt_image * _mask
            render = render * _mask
            if render_aug is not None:
                render_aug = render_aug * _mask
        rgb_diff_loss = self.rgb_diff_loss_fn(render_aug or render, gt_image)
        ssim_metric = self.ssim(render, gt_image)

        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1.0 - ssim_metric)
        metrics = {"loss": loss, "rgb_diff": rgb_diff_loss, "ssim": ssim_metric}
        prog_bar = {"loss": True, "rgb_diff": True, "ssim": True}

        # auxiliary losses
        if self.config.lambda_dreg > 0:
            scales = outputs["scales"]
            if scales.shape[0] > 0:
                scaling_reg = torch.prod(scales, dim=-1).mean()
            else:
                scaling_reg = torch.tensor(0.0)

            metrics["loss"] += self.config.lambda_dreg * scaling_reg
            metrics["loss_dreg"] = scaling_reg
            prog_bar["loss_dreg"] = False

        # pgsr flatten loss (lambda_normal > 0 means using pgsr depth rendering)
        if self.render_depth and self.config.lambda_flatten > 0:
            scales = outputs["scales"]
            if scales.shape[0] > 0:
                flatten_reg = torch.min(scales, dim=-1).values.mean()
            else:
                flatten_reg = torch.tensor(0.0)

            metrics["loss"] += self.config.lambda_flatten * flatten_reg
            metrics["loss_flatten"] = flatten_reg
            prog_bar["loss_flatten"] = False

        if self.render_depth and self.config.lambda_normal > 0 and global_step >= self.config.normal_from_iter:
            normal = outputs["normal"]
            normal_from_depth = outputs["normal_from_depth"]

            if self.config.grad_weighted_normal:
                if len(normal.shape) <= 3:
                    normal = normal.unsqueeze(0)
                    normal_from_depth = normal_from_depth.unsqueeze(0)

                # calculate plane confidence from gt_image grad
                rgb_grad = gt_image.new_ones((gt_image.shape[0], *gt_image.shape[-2:]))
                grad_x = (gt_image[..., 1:-1, 2:] - gt_image[..., 1:-1, :-2]).abs().mean(dim=1, keepdim=True)
                grad_y = (gt_image[..., 2:, 1:-1] - gt_image[..., :-2, 1:-1]).abs().mean(dim=1, keepdim=True)
                grad = torch.cat([grad_x, grad_y], dim=1).max(dim=1).values
                grad = (grad - grad.min()) / (grad.max() - grad.min())
                rgb_grad[..., 1:-1, 1:-1] = grad
                conf = (1.0 - rgb_grad) ** 2
                # loss_normal = ((normal - normal_from_depth).abs().sum(1) * conf).mean()
                loss_normal = ((1.0 - F.cosine_similarity(normal, normal_from_depth, dim=1)) * conf).mean()
            else:
                # loss_normal = ((normal - normal_from_depth).abs().sum(1)).mean()
                loss_normal = (1.0 - F.cosine_similarity(normal, normal_from_depth, dim=1)).mean()

            metrics["loss"] += self.config.lambda_normal * loss_normal
            metrics["loss_normal"] = loss_normal
            prog_bar["loss_normal"] = False

        gt_feature = extra_data.get(SemanticData.KEY, None)
        if gt_feature is not None and outputs.get("aligned_feature", None) is not None:
            aligned_feature = outputs["aligned_feature"]
            if len(aligned_feature.shape) == 3:
                aligned_feature = aligned_feature.unsqueeze(0)

            gt_feature = torch.stack(gt_feature, dim=0)
            resized = F.interpolate(
                gt_feature.permute(0, 3, 1, 2), size=aligned_feature.shape[-2:], mode="bilinear", align_corners=True
            )
            # loss_feature = 1.0 - F.cosine_similarity(aligned_feature, resized, dim=1).mean()
            loss_feature = F.l1_loss(aligned_feature, resized)

            metrics["loss"] += self.config.lambda_feature(global_step) * loss_feature
            metrics["loss_feature"] = loss_feature
            prog_bar["loss_feature"] = False

        gt_depth = extra_data.get(DepthData.KEY, None)
        if self.render_depth and gt_depth is not None:
            pred_depth = outputs["inverse_depth"]
            loss_depth = self.config.depth_loss_func(gt_depth, pred_depth, mask)

            metrics["loss"] += self.config.lambda_depth(global_step) * loss_depth
            metrics["loss_depth"] = loss_depth
            prog_bar["loss_depth"] = False

        return metrics, prog_bar

    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs):
        metrics, pbar = self._get_basic_metrics(pl_module, gaussian_model, batch, outputs)
        if (
            pl_module.trainer.datamodule.hparams["multiview"] is False
            and "pseudo_view" not in pl_module.hparams["renderer_output_types"]
        ):
            setattr(pl_module, "_current_metrics", metrics)
            return metrics, pbar

        cameras, (image_name, gt_image, _), extra_data = batch
        if isinstance(image_name, str):
            gt_image = gt_image.unsqueeze(0)
        if len(cameras.camera_center.shape) == 1:  # Camera
            params = {}
            for field in InstantiatedCameras.__dataclass_fields__:
                val = getattr(cameras, field)
                if isinstance(val, torch.Tensor):
                    val = val.unsqueeze(0)
                params[field] = val
            cameras = InstantiatedCameras(**params)

        # if step >= self.config.multiview_from_iter:
        #     rgb, depth = outputs["render"], outputs["acc_depth"]

        #     gt_l, gt_r = gt_image[0::2, ...], gt_image[1::2, ...]
        #     cam_l, cam_r = cameras[0::2], cameras[1::2]
        #     rgb_l, rgb_r = rgb[0::2, ...], rgb[1::2, ...]
        #     depth_l, depth_r = depth[0::2, ...], depth[1::2, ...]

        #     view_l = (cam_l, gt_l, rgb_l, depth_l)
        #     view_r = (cam_r, gt_r, rgb_r, depth_r)

        #     loss_multiview_l = self.multiview_loss(view_l, view_r)
        #     loss_multiview_r = self.multiview_loss(view_r, view_l)
        #     loss_multiview = (loss_multiview_l + loss_multiview_r) / 2.0

        #     metrics["loss"] += self.config.lambda_multiview(step) * loss_multiview
        #     metrics["loss_multiview"] = loss_multiview
        #     pbar["loss_multiview"] = False

        # patch based multiview loss
        # if step >= self.config.multiview_from_iter:
        #     pseudo_results = outputs.get("pseudo_results", None)
        #     if pseudo_results is not None:
        #         render_ps = pseudo_results["render"]
        #         cameras_ps = pseudo_results["view"]
        #         loss_multiview_dict = self.config.multiview_loss_func.patch_loss(outputs, render_ps, cameras, cameras_ps, gt_image)
        #         loss_multiview = loss_multiview_dict.pop("loss_multiview", 0.0)
        #         metrics["loss"] += loss_multiview

        #         for k, v in loss_multiview_dict:
        #             metrics[k] = v
        #             pbar[k] = False

        if step >= self.config.multiview_from_iter:
            pseudo_results = outputs.get("pseudo_results", None)
            if pseudo_results is not None:
                n_cam = len(cameras)
                rgb, depth, inv_depth = outputs["render"], outputs["acc_depth"], outputs["inverse_depth"]
                rgb, depth, inv_depth = tuple(
                    map(lambda x: x.unsqueeze(0) if n_cam == 1 else x, [rgb, depth, inv_depth])
                )
                outputs_ps, cameras_ps = outputs["pseudo_results"]["render"], outputs["pseudo_results"]["view"]
                rgb_ps, depth_ps = outputs_ps["render"], outputs_ps["acc_depth"]
                rgb_ps, depth_ps = tuple(map(lambda x: x.unsqueeze(0) if n_cam == 1 else x, [rgb_ps, depth_ps]))

                points2d_ndc, mask, pixel_shift = MultiView.symmetric_transformation(
                    cameras, cameras_ps, depth, depth_ps
                )
                mask = mask & (pixel_shift < 1.0).clone().detach().bool()
                # mask = mask.unsqueeze(1)
                warp_rgb = F.grid_sample(rgb_ps, points2d_ndc, align_corners=True)

                num_pixels = mask.sum(dim=[-1, -2])
                # loss_pixshift = (pixel_shift * mask).sum(dim=[-1, -2])
                # loss_pixshift = (loss_pixshift / (num_pixels + 1e-8)).mean()
                # loss_multiview = self.rgb_diff_loss_fn(gt_image * mask, warp_rgb)
                loss_multiview = ((warp_rgb - rgb).abs().mean(dim=1) * mask).sum(dim=[-1, -2])
                loss_multiview = (loss_multiview / (num_pixels + 1e-8)).mean()

                # metrics["loss_pixshift"] = loss_pixshift
                # pbar["loss_pixshift"] = False
                metrics["loss_multiview"] = loss_multiview
                pbar["loss_multiview"] = False
                metrics["loss"] += self.config.lambda_multiview(step) * loss_multiview

                if step % 2000 == 0:
                    image_tensor = torch.stack([rgb_ps, warp_rgb, warp_rgb * mask.unsqueeze(1), gt_image], dim=1)
                    grid = torchvision.utils.make_grid(image_tensor.reshape(-1, *rgb.shape[1:]), 4)
                    pl_module.log_image(tag="pseudo_view", image_tensor=grid)

        setattr(pl_module, "_current_metrics", metrics)
        return metrics, pbar

    def multiview_loss(self, view_l, view_r):
        cam_l, gt_l, rgb_l, depth_l = view_l
        cam_r, gt_r, rgb_r, depth_r = view_r
        points3d = depth_to_points(
            depth_l.permute(0, 2, 3, 1),
            cam_l.world_to_camera.transpose(-1, -2).inverse(),
            MultiView.get_Ks(cam_l),
            True,
        )
        points2d, mask = MultiView.reproject(points3d, cam_r)
        warp_rgb = F.grid_sample(gt_r, points2d, align_corners=True)
        loss_multiview = (((warp_rgb - rgb_l)).abs().mean(dim=1) * mask).sum(dim=[-1, -2]) / mask.sum(dim=[-1, -2])
        return loss_multiview.mean()
