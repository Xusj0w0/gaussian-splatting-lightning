from dataclasses import dataclass, field
from internal.metrics.metric import Metric, MetricImpl
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class AppearanceModelingMetrics(VanillaMetrics):
    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return AppearanceModelingMetricsImpl(self)


class AppearanceModelingMetricsImpl(VanillaMetricsImpl):
    def _get_basic_metrics(self, pl_module, gaussian_model, batch, outputs):
        camera, image_info, _ = batch
        image_name, gt_image, masked_pixels = image_info
        image = outputs["render"]
        image_augmented = outputs.get("appearance_augmented", image)

        # calculate loss
        if masked_pixels is not None:
            gt_image = gt_image.clone()
            gt_image[masked_pixels] = image.detach()[masked_pixels]  # copy masked pixels from prediction to G.T.

        rgb_diff_loss = self.rgb_diff_loss_fn(image_augmented, gt_image)
        ssim_metric = self.ssim(image, gt_image)

        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1.0 - ssim_metric)

        return {
            "loss": loss,
            "rgb_diff": rgb_diff_loss,
            "ssim": ssim_metric,
        }, {
            "loss": True,
            "rgb_diff": True,
            "ssim": True,
        }

