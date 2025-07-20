import json
import os
from math import exp

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from .lpipsPyTorch.modules.lpips import LPIPS

def lpips(x: torch.Tensor, y: torch.Tensor, net_type: str = "alex", version: str = "0.1"):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
    """Warp `img` to match the colors in `ref_img`."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match")
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c : (c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(jnp.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = jnp.concatenate(a_mat, axis=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = jnp.where(mask[:, None], a_mat, 0)
            mb = jnp.where(mask, b, 0)
            # Solve the linear system. We're using the np.lstsq instead of jnp because
            # it's significantly more stable in this case, for some reason.
            w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert jnp.all(jnp.isfinite(w))
            warp.append(w)
        warp = jnp.stack(warp, axis=-1)
        # Apply the warp to update img_mat.
        img_mat = jnp.clip(jnp.matmul(a_mat, warp, precision=jax.lax.Precision.HIGHEST), 0, 1)
    corrected_img = jnp.reshape(img_mat, img.shape)
    return corrected_img


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
