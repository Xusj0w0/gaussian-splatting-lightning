from dataclasses import dataclass

import torch
import torch.nn.functional as F

from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl

__all__ = ["FeatureMetrics", "FeatureMetricsImpl"]


@dataclass
class FeatureMetrics(VanillaMetrics):
    lambda_dreg: float = 0.01

    lambda_feature: float = 0.5

    lambda_normal: float = 0.0

    lambda_dist: float = 0.0

    normal_start_iter: int = 7_000

    dist_start_iter: int = 3_000

    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs):
        return FeatureMetricsImpl(self)


class FeatureMetricsImpl(VanillaMetricsImpl):
    config: FeatureMetrics

    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)

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

        if self.config.lambda_feature > 0:
            render_feature = outputs["render_feature_aligned"]
            gt_feature = batch[-1]["semantic_feature"]
            # loss_feature = F.mse_loss(render_feature, gt_feature)
            # loss_feature = 1.0 - F.cosine_similarity(render_feature, gt_feature, dim=-1).mean()
            loss_feature = F.l1_loss(render_feature, gt_feature, reduction="mean")
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
