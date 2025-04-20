from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from myimpl.model_components.feature_adapter import AdapterConfig

__all__ = ["FeatureMetrics", "FeatureMetricsImpl"]


@dataclass
class FeatureMetrics(VanillaMetrics):
    lambda_dreg: float = 0.01

    lambda_feature: float = 0.5

    lambda_normal: float = 0.0

    lambda_dist: float = 0.0

    feature_start_iter: Optional[int] = None

    feature_end_iter: Optional[int] = None

    normal_start_iter: int = 7_000

    dist_start_iter: int = 3_000

    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs):
        return FeatureMetricsImpl(self)


class FeatureMetricsImpl(VanillaMetricsImpl):
    config: FeatureMetrics

    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)

        if stage == "fit":
            if self.config.feature_start_iter is None:
                self.config.feature_start_iter = 0
            if self.config.feature_end_iter is None:
                self.config.feature_end_iter = pl_module.trainer.max_steps

            # hwc shape
            gt_feature_shape = pl_module.trainer.datamodule.dataparser_outputs.train_set.extra_data[0]["semantic_feature"].shape  # fmt: skip
            self.output_size = gt_feature_shape[:2]

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

        if (
            self.config.lambda_feature > 0
            and global_step >= self.config.feature_start_iter
            and global_step <= self.config.feature_end_iter + 1
        ):
            adapted_render_feature = outputs["render_feature_adapted"]
            gt_feature = batch[-1]["semantic_feature"]

            # fmt: off
            adapted_render_feature = F.interpolate(
                adapted_render_feature.permute(2, 0, 1).unsqueeze(0), size=self.output_size, mode="bicubic", align_corners=True
            ).squeeze(0).permute(1, 2, 0)
            # fmt: on

            loss_feature = F.l1_loss(adapted_render_feature, gt_feature)

            metrics["loss"] += self.config.lambda_feature * loss_feature
            metrics["loss_feature"] = loss_feature
            prog_bar["loss_feature"] = True

        if self.config.lambda_normal > 0 and global_step >= self.config.normal_start_iter:
            # key names from `vanilla_2dgs_renderer`
            rend_normals = outputs["rend_normal"]
            surf_normals = outputs["surf_normal"]
            loss_normal = (1 - (rend_normals * surf_normals).sum(dim=-1)).mean()
            metrics["loss"] += self.config.lambda_normal * loss_normal
            metrics["loss_normal"] = loss_normal
            prog_bar["loss_normal"] = True

        if self.config.lambda_dist > 0 and global_step >= self.config.dist_start_iter:
            loss_dist = outputs["rend_dist"].mean()
            metrics["loss"] += self.config.lambda_dist * loss_dist
            metrics["loss_dist"] = loss_dist
            prog_bar["loss_dist"] = True

        return metrics, prog_bar
