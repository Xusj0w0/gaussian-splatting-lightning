from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from internal.cameras.cameras import Camera
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from myimpl.utils.dataset_utils import (DepthData, ExtraDataProcessorOutputs,
                                        MaskData, SemanticData)

__all__ = ["ScaffoldMetrics", "ScaffoldMetricsImpl"]


@dataclass
class ScaffoldMetrics(VanillaMetrics):
    lambda_dreg: float = 0.01

    lambda_flatten: float = 100.0

    lambda_normal: float = 0.015

    normal_start_iter: int = 7_000

    grad_weighted_normal: bool = True

    lambda_depth: "WeightScheduler" = field(default_factory=lambda: WeightScheduler())

    depth_loss_func: "DepthLossFunction" = field(default_factory=lambda: DepthLossFunction())

    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs):
        return ScaffoldMetricsImpl(self)


class ScaffoldMetricsImpl(VanillaMetricsImpl):
    config: ScaffoldMetrics

    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)

        if self.config.lambda_depth.max_steps is None:
            self.config.lambda_depth.max_steps = pl_module.trainer.max_steps
        self.render_depth = "acc_depth" in pl_module.renderer_output_types

    def _get_basic_metrics(
        self,
        pl_module,
        gaussian_model,
        batch: Tuple[Camera, Tuple[str, torch.Tensor, Optional[torch.Tensor]], ExtraDataProcessorOutputs],
        outputs,
    ):
        metrics, prog_bar = super()._get_basic_metrics(pl_module, gaussian_model, batch, outputs)
        global_step = pl_module.trainer.global_step + 1
        _, image_info, extra_data = batch
        image_name, gt_image, mask = image_info
        if isinstance(image_name, str):
            extra_data = {k: [v] for k, v in extra_data.items()}

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

        if self.render_depth and self.config.lambda_normal > 0 and global_step >= self.config.normal_start_iter:
            normal_map = outputs["normal"]
            normal_map_from_depth = outputs["normal_from_depth"]

            if self.config.grad_weighted_normal:
                # calculate plane confidence from gt_image grad
                rgb_grad = gt_image.new_ones(gt_image.shape[-2:])
                grad_x = (gt_image[..., 1:-1, 2:] - gt_image[..., 1:-1, :-2]).abs().mean(dim=0, keepdim=True)
                grad_y = (gt_image[..., 2:, 1:-1] - gt_image[..., :-2, 1:-1]).abs().mean(dim=0, keepdim=True)
                grad = torch.cat([grad_x, grad_y], dim=0).max(dim=0).values
                grad = (grad - grad.min()) / (grad.max() - grad.min())
                rgb_grad[1:-1, 1:-1] = grad
                conf = (1.0 - rgb_grad) ** 2
                loss_normal = ((normal_map - normal_map_from_depth).abs().sum(-1) * conf).mean()
            else:
                loss_normal = ((normal_map - normal_map_from_depth).abs().sum(-1)).mean()

            metrics["loss"] += self.config.lambda_normal * loss_normal
            metrics["loss_normal"] = loss_normal
            prog_bar["loss_normal"] = False

        gt_depth = extra_data.get(DepthData.KEY, None)
        if self.render_depth and gt_depth is not None:
            pred_depth = outputs["inverse_depth"]
            loss_depth = self.config.depth_loss_func(gt_depth, pred_depth, mask)

            metrics["loss"] += self.config.lambda_depth(global_step) * loss_depth
            metrics["loss_depth"] = loss_depth
            prog_bar["loss_depth"] = False

        return metrics, prog_bar


@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: Optional[int] = 30_000

    def __call__(self, step: int) -> float:
        return self.init * (self.final_factor ** min(step / self.max_steps, 1.0))


@dataclass
class DepthLossFunction:
    type: Literal["l1", "l2", "kl"] = "l1"
    """Type of depth loss function."""

    ssim_weight: float = 0.2

    normalized: bool = False

    median_normalized: bool = False

    mean_normalized: bool = False

    def __call__(
        self, gt_depth: List[torch.Tensor], pred_depth: torch.Tensor, mask: Optional[List[torch.Tensor]] = None
    ):
        loss = torch.tensor(0.0, device=pred_depth.device)
        cnt = 0
        for i in range(len(gt_depth)):
            gt = gt_depth[i]

            if gt is not None:
                msk = mask[i] if mask is not None else None
                loss += self.loss_iter(gt, pred_depth[i], msk)
                cnt += 1
        return loss / cnt

    def loss_iter(
        self, gt_depth: torch.Tensor, pred_depth: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        gt_depth = gt_depth.to(pred_depth)
        if self.normalized:
            with torch.no_grad():
                max_depth = pred_depth.max()
                min_depth = pred_depth.min()
            pred_depth = (pred_depth - min_depth) / (max_depth - min_depth + 1e-8)
        elif self.median_normalized:
            median = torch.median(gt_depth)
            gt_depth = gt_depth / median
            pred_depth = pred_depth / median
        elif self.mean_normalized:
            mean = torch.mean(gt_depth)
            gt_depth = gt_depth / mean
            pred_depth = pred_depth / mean

        if mask is not None:
            mask = mask.to(pred_depth)
            gt_depth = gt_depth * mask
            pred_depth = pred_depth * mask

        return self.calc_loss(gt_depth, pred_depth)

    def calc_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.type == "l1":
            return torch.abs(a - b).mean()
        elif self.type == "l2":
            return ((a - b) ** 2).mean()
        elif self.type == "kl":
            raise NotImplementedError("KL divergence loss is not implemented.")
        else:
            raise ValueError(f"Unknown depth loss type: {self.type}")

    def _depth_l1_loss(self, a, b):
        return torch.abs(a - b).mean()

    def _depth_l1_and_ssim_loss(self, a, b):
        l1_loss = self._depth_l1_loss(a, b)
        ssim_metric = self.depth_ssim(a[None, None, ...], b[None, None, ...])

        return (1 - self.ssim_weight) * l1_loss + self.ssim_weight * (1 - ssim_metric)

    def _depth_l2_loss(self, a, b):
        return ((a - b) ** 2).mean()

    def _depth_kl_loss(self, a, b):
        pass
