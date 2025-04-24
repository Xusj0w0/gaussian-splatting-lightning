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
class WeightScheduler:
    init: float = -1.0
    """init weight of loss depth. <0 means no depth supervision"""

    final_factor: float = 0.01

    max_steps: Optional[int] = 30_000


@dataclass
class DepthLossConfig:
    type: Literal["l1", "l2", "kl"] = "l1"
    """Type of depth loss function."""

    ssim_weight: float = 0.2

    normalized: bool = False

    median_normalized: bool = False

    mean_normalized: bool = False


@dataclass
class ScaffoldMetrics(VanillaMetrics):
    lambda_dreg: float = 0.01

    lambda_flatten: float = 100.0

    lambda_normal: float = 0.0

    lambda_depth_scheduler: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    depth_loss_config: DepthLossConfig = field(default_factory=lambda: DepthLossConfig())

    normal_start_iter: int = 7_000

    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs):
        return ScaffoldMetricsImpl(self)


class ScaffoldMetricsImpl(VanillaMetricsImpl):
    config: ScaffoldMetrics

    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)

        if self.config.lambda_depth_scheduler.max_steps is None:
            self.config.lambda_depth_scheduler.max_steps = pl_module.trainer.max_steps

    def _get_basic_metrics(self, pl_module, gaussian_model, batch, outputs):
        metrics, prog_bar = super()._get_basic_metrics(pl_module, gaussian_model, batch, outputs)

        global_step = pl_module.trainer.global_step + 1

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
        if self.config.lambda_flatten > 0 and self.config.lambda_normal > 0:
            scales = outputs["scales"]
            if scales.shape[0] > 0:
                flatten_reg = torch.min(scales, dim=-1).values.means()
            else:
                flatten_reg = torch.tensor(0.0)
            metrics["loss"] += self.config.lambda_flatten * flatten_reg
            metrics["loss_flatten"] = flatten_reg
            prog_bar["loss_flatten"] = False

        if self.config.lambda_normal > 0 and global_step >= self.config.normal_start_iter:
            normal_map = outputs["normals"]
            normal_map_from_depth = outputs["normals_from_depths"]
            loss_normal = (1.0 - (normal_map * normal_map_from_depth).sum(-1)).mean()

            metrics["loss"] += self.config.lambda_normal * loss_normal
            metrics["loss_normal"] = loss_normal
            prog_bar["loss_normal"] = False

        lambda_depth = self.config.lambda_depth_scheduler(global_step)
        if lambda_depth > 0:
            batch: Tuple[Camera, Tuple[str, torch.Tensor, Optional[torch.Tensor]], ExtraDataProcessorOutputs]
            _, _, extra_data = batch
            gt_depth = extra_data.get(DepthData.KEY, None)
            if gt_depth is not None:
                pred_depth = outputs["inverse_depths"]
                mask = extra_data.get(MaskData.KEY, None)
                loss_depth = self.config.depth_loss_func(gt_depth, pred_depth, mask)

                metrics["loss"] += lambda_depth * loss_depth
                metrics["loss_depth"] = loss_depth
                prog_bar["loss_depth"] = False

        return metrics, prog_bar

    @classmethod
    def get_loss_weight(cls, weight_scheduler: "WeightScheduler"):
        if weight_scheduler.init < 0:
            return -1
        return weight_scheduler.init * (weight_scheduler.final_factor ** min(weight_scheduler.max_steps, 1.0))

    def get_depth_loss(
        self, gt_depth: torch.Tensor, pred_depth: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        gt_depth = gt_depth.to(pred_depth)

        if self.config.depth_loss_config.normalized:
            with torch.no_grad():
                max_depth = pred_depth.max()
                min_depth = pred_depth.min()
            pred_depth = (pred_depth - min_depth) / (max_depth - min_depth + 1e-8)
        elif self.config.depth_loss_config.median_normalized:
            median = torch.median(gt_depth)
            gt_depth = gt_depth / median
            pred_depth = pred_depth / median
        elif self.config.depth_loss_config.mean_normalized:
            mean = torch.mean(gt_depth)
            gt_depth = gt_depth / mean
            pred_depth = pred_depth / mean

        if mask is not None:
            mask = mask.to(pred_depth)
            gt_depth = gt_depth * mask
            pred_depth = pred_depth * mask

        if self.config.depth_loss_config.type == "l1":
            return torch.abs(gt_depth - pred_depth).mean()
        elif self.config.depth_loss_config.type == "l2":
            return ((gt_depth - pred_depth) ** 2).mean()
        elif self.config.depth_loss_config.type == "kl":
            raise NotImplementedError("KL divergence loss is not implemented.")
        else:
            raise ValueError(f"Unknown depth loss type: {self.config.depth_loss_config.type}")
