import torch
from gsplat.rasterize_to_weighted_alphas import rasterize_to_weighted_alphas

from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule)
import torch.nn.functional as F

class ImGradMetrics(VanillaMetrics):
    def instantiate(self, *args, **kwargs):
        return ImGradMetricsImpl(self)


class ImGradMetricsImpl(VanillaMetricsImpl):
    config: ImGradMetrics

    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs):
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        camera, image_info, _ = batch
        image_name, gt_image, masked_pixels = image_info

        # _, _, (image_width, image_height) = GSplatV1.preprocess_camera(camera)
        # means2d = outputs["viewspace_points"]
        # conics = outputs["conics"]
        # opacities = outputs["opacities"]
        # _, _, flatten_ids, isect_offsets = outputs["isects"]

        # imgrad = self.compute_image_grad(gt_image)
        # if masked_pixels is not None:
        #     imgrad = imgrad * masked_pixels[0:1, ...]
        # accum_weights = rasterize_to_weighted_alphas(
        #     means2d=means2d.unsqueeze(0),
        #     conics=conics,
        #     opacities=opacities.unsqueeze(0),
        #     image_width=image_width,
        #     image_height=image_height,
        #     tile_size=pl_module.renderer.config.block_size,
        #     isect_offsets=isect_offsets,
        #     flatten_ids=flatten_ids,
        #     pixel_weights=imgrad,
        # )[0]
        # means2d.imgrad = accum_weights.squeeze()

        image = outputs["render"]
        grad_x, grad_y = self.apply_sobel(image, direction='x'), self.apply_sobel(image, direction='y')
        gt_grad_x, gt_grad_y = self.apply_sobel(gt_image, direction='x'), self.apply_sobel(gt_image, direction='y')
        if masked_pixels is not None:
            grad_x, grad_y, gt_grad_x,gt_grad_y = map(lambda x: x * masked_pixels[0:1, ...], [grad_x, grad_y, gt_grad_x, gt_grad_y])
        
        factor = 1.0 / (image.shape[-1] * image.shape[-2]) ** 0.5
        diff_grad_x = torch.abs(grad_x - gt_grad_x)
        diff_grad_y = torch.abs(grad_y - gt_grad_y)
        loss_grad = 
        metrics["loss_grad"] = 

        return metrics, pbar

    def compute_image_grad(self, image: torch.Tensor):
        if len(image.shape) <= 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape

        imgrad = image.new_zeros(B, H, W)
        grad_x = (image[..., 1:-1, 2:] - image[..., 1:-1, :-2]).abs().mean(dim=1, keepdim=True)
        grad_y = (image[..., 2:, 1:-1] - image[..., :-2, 1:-1]).abs().mean(dim=1, keepdim=True)
        grad = torch.cat([grad_x, grad_y], dim=1).max(dim=1).values
        imgrad[..., 1:-1, 1:-1] = grad
        return imgrad

    def apply_sobel(self, image:torch.Tensor, direction:str='x'):
        if direction == 'x':
            kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(image).unsqueeze(0).unsqueeze(0)
        elif direction == 'y':
            kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(image).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Direction must be 'x' or 'y'.")
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        return F.conv2d(image, kernel, padding=1)