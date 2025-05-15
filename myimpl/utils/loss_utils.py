from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from gsplat.utils import depth_to_points
from torchmetrics.image import StructuralSimilarityIndexMeasure

from internal.cameras.cameras import Camera, Cameras
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from myimpl.utils.cameras import InstantiatedCameras


@dataclass
class WeightScheduler:
    enabled: bool = False

    init: float = 1.0

    final_factor: float = 0.01

    mode: Literal["log", "exp", "linear"] = "linear"

    max_steps: Optional[int] = None

    def __call__(self, step: int) -> float:
        t = np.clip(step / self.max_steps, 0.0, 1.0)
        if self.mode == "linear":
            return self.init * (1.0 - t) + self.init * self.final_factor * t
        elif self.mode == "exp":
            return self.init * (self.final_factor**t)
        elif self.mode == "log":
            return np.exp(np.log(self.init) * (1.0 - t) + np.log(self.init * self.final_factor) * t)
        else:
            raise ValueError(f"unsupported mode")


@dataclass
class DepthRegularization:
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


class MultiView:
    @classmethod
    @torch.no_grad()
    def get_pseudo_view(cls, cameras: Cameras, depth: torch.Tensor, disturb: float):
        """
        :Args:
        - **cameras: Cameras**
        - **depth: [C, 1, H, W]**
        - **disturb**: float, the std of the noise

        :Returns:
        - **pseudo_cameras**: Cameras
        """
        n_cam = len(cameras)
        rel_pos = torch.normal(0.0, 1.0, (n_cam, 3)).to(cameras.R) * disturb
        # alpha = torch.rand((n_cam,)).to(cameras.R) * 2 * torch.pi
        # rel_pos = torch.zeros((n_cam, 3)).to(cameras.R)
        # rel_pos[:, 0] = disturb * torch.cos(alpha)
        # rel_pos[:, 2] = disturb * torch.sin(alpha)
        rel_pos[:, 1] = 0.0  # don't disturb in y-axis

        median_depth = torch.median(depth.flatten(1), dim=-1, keepdim=True).values
        center = torch.cat([median_depth.new_zeros((n_cam, 2)), median_depth], dim=-1)  # [C, 3]
        y_axis = center.new_zeros((n_cam, 3))
        y_axis[:, 1] = 1.0
        z_axis = F.normalize(center - rel_pos, dim=-1)
        x_axis = F.normalize(torch.cross(y_axis, z_axis, dim=-1), dim=-1)
        rel_rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [C, 3, 3]

        # pseudo_to_ref transformation
        pseudo_to_ref = rel_rot.new_zeros((n_cam, 4, 4))
        pseudo_to_ref[:, :3, :3] = rel_rot
        pseudo_to_ref[:, :3, -1] = rel_pos
        pseudo_to_ref[:, -1, -1] = 1.0

        # world_to_ref
        world_to_ref = cameras.world_to_camera.transpose(-1, -2)  # cameras.world_to_camera is transposed

        # world_to_pseudo = ref_to_pseudo @ world_to_ref
        world_to_pseudo = torch.einsum("...ij, ...jk -> ...ik", pseudo_to_ref.inverse(), world_to_ref)

        params = {k: getattr(cameras, k) for k, v in Cameras.__dataclass_fields__.items() if v.init}
        params.update({"R": world_to_pseudo[:, :3, :3], "T": world_to_pseudo[:, :3, -1]})

        pseudo_cameras = Cameras(**params)
        for k in Cameras.__dataclass_fields__:
            val_pseudo = getattr(pseudo_cameras, k)
            val = getattr(cameras, k)
            if val.device != val_pseudo.device:
                setattr(pseudo_cameras, k, val_pseudo.to(val))
        return pseudo_cameras

    @classmethod
    def get_Ks(cls, camera: Cameras):
        """
        :Args:
        - **camera: Cameras**

        :Returns:
        - **Ks**: [C, 3, 3], the intrinsic matrix of each camera
        """
        Ks = torch.zeros((camera.R.shape[0], 3, 3), dtype=torch.float, device=camera.R.device)
        Ks[..., 0, 0] = camera.fx
        Ks[..., 1, 1] = camera.fy
        Ks[..., 0, 2] = camera.cx
        Ks[..., 1, 2] = camera.cy
        Ks[..., 2, 2] = 1.0
        return Ks

    @classmethod
    def reproject(cls, points3d: torch.Tensor, cameras: Cameras):
        """
        :Args:
        - **points3d**: [C, H, W, 3], pointmap in world coordinates
        - **cameras**: Cameras

        :Returns:
        - **points2d**: [C, H, W, 2]
        - **points2d_ndc**: [C, H, W, 2], in [-1, 1]x[-1, 1]
        - **mask**: [C, H, W]
        """
        n_cam, H, W = points3d.shape[:-1]
        assert (
            len(cameras) == n_cam
        ), f"the number of cameras in cameras_ref and cameras_qry must be the same, but got {n_cam} and {len(cameras)}"

        points3d_cam = cameras.world_to_camera[..., -1:, :3] + torch.einsum(
            "...dm, ...nd -> ...nm", cameras.world_to_camera[..., :3, :3], points3d.reshape(n_cam, -1, 3)
        )  # [C, H*W, 3]

        points2d = points3d_cam.clone()
        points2d /= points3d_cam[..., -1:] + 1e-8  # [C, H, W, 3]

        # get points2d in pixel coordinates
        Ks = cls.get_Ks(cameras)
        points2d_pix = torch.einsum("...md, ...nd -> ...nm", Ks, points2d)[..., :2]

        points2d_ndc = points2d_pix[..., :2].clone()
        scale_x, scale_y = 2.0 / (cameras.width[..., None] - 1.0), 2.0 / (cameras.height[..., None] - 1.0)
        points2d_ndc[..., 0] = (points2d_ndc[..., 0] - 0.5) * scale_x - 1.0
        points2d_ndc[..., 1] = (points2d_ndc[..., 1] - 0.5) * scale_y - 1.0

        mask = (
            (points3d_cam[..., -1] > 1e-2)
            & torch.prod(points2d_ndc > -1, dim=-1)
            & torch.prod(points2d_ndc < 1, dim=-1)
        )

        points2d = points2d[..., :2].reshape(n_cam, H, W, 2)
        points2d_ndc = points2d_ndc.reshape(n_cam, H, W, 2)
        mask = mask.reshape(n_cam, H, W)
        return points2d[..., :2], points2d_ndc, mask

    @classmethod
    def symmetric_transformation(
        cls, ref_cams: Cameras, query_cams: Cameras, ref_depth: torch.Tensor, query_depth: torch.Tensor
    ):
        """
        :Args:
        - **ref_cams**: Cameras
        - **query_cams**: Cameras
        - **ref_depth**: [C, 1, H, W]
        - **query_depth**: [C, 1, H, W]

        :Returns:
        - **points2d_ndc**: [C, H, W, 2], in [-1, 1] x [-1, 1]. pointmap of reference view projected to query view
        - **mask**: [C, H, W]
        - **pixel_shift**: [C, H, W]
        """
        n_cam = len(ref_cams)
        height, width = ref_depth.shape[-2:]

        # ref to pseudo
        points3d = depth_to_points(
            ref_depth.permute(0, 2, 3, 1),
            ref_cams.world_to_camera.transpose(-1, -2).inverse(),
            cls.get_Ks(ref_cams),
            True,
        )
        points2d, points2d_ndc, mask = cls.reproject(points3d, query_cams)

        # pseudo back to ref
        depth_ = F.grid_sample(query_depth, points2d_ndc, align_corners=True)
        points3d_ = depth_.permute(0, 2, 3, 1) * torch.cat(
            [points2d, points2d.new_ones((*points2d.shape[:-1], 1))], dim=-1
        )
        c2w = query_cams.world_to_camera.transpose(-1, -2).inverse()
        points3d_ = c2w[..., :3, -1].unsqueeze(1).unsqueeze(1) + torch.einsum(
            "...md, ...hwd -> ...hwm", c2w[..., :3, :3], points3d_
        )
        _, points2d_ndc_, _ = cls.reproject(points3d_, ref_cams)
        points2d_pix_ = points2d_ndc_.clone()
        points2d_pix_[..., 0] = (points2d_pix_[..., 0] + 1.0) * (float(width) - 1) / 2
        points2d_pix_[..., 1] = (points2d_pix_[..., 1] + 1.0) * (float(height) - 1) / 2

        coordinates = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy"), dim=-1)
        pixel_shift = (points2d_pix_ - coordinates.to(points2d_pix_)).norm(dim=-1)
        return points2d_ndc, mask, pixel_shift


@dataclass
class PatchMultiviewLoss:
    pixel_diff_thresh: float = 1.0

    lambda_geom: float = 0.03

    lambda_ncc: float = 0.15

    patch_size: int = 7

    patch_sample_num: int = 2 << 16

    check_plane_hypothesis: bool = True

    def reproj_loss(
        self,
        output_pkg: Dict[str, torch.Tensor],
        output_pkg_pseudo: Dict[str, torch.Tensor],
        cameras: Cameras,
        cameras_pseudo: Cameras,
        rgb_gt: Optional[torch.Tensor] = None,
    ):
        n_cam = len(cameras)

        rgb, depth, inv_depth = output_pkg["render"], output_pkg["inverse_depth"], output_pkg["acc_depth"]
        rgb, depth, inv_depth = tuple(map(lambda x: x.unsqueeze(0) if n_cam == 1 else x, [rgb, depth, inv_depth]))
        rgb_ps, depth_ps = output_pkg_pseudo["render"], output_pkg_pseudo["acc_depth"] # fmt: skip
        rgb_ps, depth_ps = tuple(map(lambda x: x.unsqueeze(0) if n_cam == 1 else x, [rgb_ps, depth_ps]))

        points3d = depth_to_points(
            depth.permute(0, 2, 3, 1),
            cameras.world_to_camera.transpose(-1, -2).inverse(),
            MultiView.get_Ks(cameras),
            True,
        )
        points2d, mask = MultiView.reproject(points3d, cameras_pseudo)
        mask = mask & (inv_depth.squeeze(1) > 1e-2)
        warp_rgb = F.grid_sample(rgb_ps, points2d, align_corners=True)

        loss_multiview = (warp_rgb - rgb_gt).sum(dim=[-1, -2]) / (mask.sum(dim=[-1, -2]) + 1e-8)
        loss_multiview = loss_multiview.mean(dim=0)
        return {"loss_multiview": loss_multiview}

    def patch_loss(
        self,
        output_pkg: Dict[str, torch.Tensor],
        output_pkg_pseudo: Dict[str, torch.Tensor],
        cameras: Cameras,
        cameras_pseudo: Cameras,
        rgb_gt: Optional[torch.Tensor] = None,
    ):
        """ """
        n_cam = len(cameras)

        rgb, depth, normal, plane_dist = (
            output_pkg["render"],
            output_pkg["acc_depth"],
            output_pkg["normal"],
            output_pkg["plane_dist"],
        )
        rgb, depth, normal = tuple(map(lambda x: x.unsqueeze(0) if n_cam == 1 else x, [rgb, depth, normal]))
        normal_local = torch.einsum(
            "...dm, ...dhw -> ...hwm", cameras.world_to_camera[..., :3, :3], normal
        )  # dm: w2c is transposed
        gray = self.rgb2gray(rgb_gt if rgb_gt is not None else rgb)

        rgb_ps, depth_ps = output_pkg_pseudo["render"], output_pkg_pseudo["acc_depth"]
        rgb_ps, depth_ps = tuple(map(lambda x: x.unsqueeze(0) if n_cam == 1 else x, [rgb_ps, depth_ps]))
        gray_ps = self.rgb2gray(rgb_ps)

        H, W = rgb.shape[-2:]

        pixels = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"), dim=-1).to(rgb) + 0.5
        pixels_, mask = MultiView.symmetric_transform(depth, depth_ps, cameras, cameras_pseudo)
        pixel_diff = torch.norm(pixels_ - pixels.unsqueeze(0), dim=-1)  # [C, H, W]

        if self.check_plane_hypothesis:
            mask = mask & (pixel_diff < self.pixel_diff_thresh)  # [C, H, W]
            weights = torch.exp(-pixel_diff).detach()
            weights = weights * mask.float()
        else:
            weights = pixel_diff.new_ones(pixel_diff.shape)
            weights = weights * mask.float()

        loss_geom, loss_ncc = [], []
        # compute multi-view loss in loop
        for cam_id in range(n_cam):
            _mask, _weights = mask[cam_id], weights[cam_id]
            if _mask.sum() <= 0:
                continue

            loss_geom.append((pixel_diff[cam_id] * _weights * _mask.float()).sum() / _mask.sum())

            # return [n_samples, patch_size, patch_size]
            gray_patch_ref, gray_patch_qry, valid_indices = self.sample_patch(
                _mask,
                gray[cam_id],
                gray_ps[cam_id],
                cameras[cam_id],
                cameras_pseudo[cam_id],
                normal_local[cam_id],
                plane_dist[cam_id],
            )
            ncc, ncc_mask = self.lncc(gray_patch_ref, gray_patch_qry)
            if ncc_mask.sum() > 0:
                loss_ncc.append((ncc * _weights.view(-1)[valid_indices])[ncc_mask].mean())

        loss_geom = torch.mean(torch.stack(loss_geom)) if len(loss_geom) > 0 else torch.tensor(0.0, device=rgb.device)
        loss_ncc = torch.mean(torch.stack(loss_ncc)) if len(loss_ncc) > 0 else torch.tensor(0.0, device=rgb.device)
        return {
            "loss_geom": loss_geom,
            "loss_ncc": loss_ncc,
            "loss_multiview": self.lambda_geom * loss_geom + self.lambda_ncc * loss_ncc,
        }

    def sample_patch(
        self,
        mask: torch.Tensor,
        gray_ref: torch.Tensor,
        gray_qry: torch.Tensor,
        cam_ref: Camera,
        cam_qry: Camera,
        normal_local: torch.Tensor,
        plane_dist: torch.Tensor,
    ):
        device = gray_qry.device
        H, W = gray_ref.shape[-2:]

        valid_indices = torch.arange(H * W, device=device)
        patch_offsets = self.patch_offsets(self.patch_size, device=device)  # [ps*ps, 2]
        pixels = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"), dim=-1).to(gray_ref) + 0.5

        with torch.no_grad():
            valid = valid_indices[mask.reshape(-1)]
            valid = valid[torch.randint(0, valid.shape[0], (self.patch_sample_num,), device=device)]

        valid_pixels = pixels.reshape(-1, 2)[valid].unsqueeze(1) + patch_offsets  # [n_samples, ps*ps, 2]
        valid_normal = normal_local.view(-1, 3)[valid]  # [n_samples, 3]
        valid_plane_dist = plane_dist.view(-1)[valid]  # [n_samples, ]

        # sample gray values
        normalized_valid_pixels = valid_pixels.clone()
        normalized_valid_pixels[..., 0] = normalized_valid_pixels[..., 0] / (float(W) / 2) - 1.0
        normalized_valid_pixels[..., 1] = normalized_valid_pixels[..., 1] / (float(H) / 2) - 1.0
        gray_patch_ref = F.grid_sample(
            gray_ref.unsqueeze(0), normalized_valid_pixels.view(1, -1, 1, 2), align_corners=True
        )  # [1, n_samples*ps*ps, 1, 1]
        gray_patch_ref = gray_patch_ref.reshape(
            self.patch_sample_num, self.patch_size, self.patch_size
        )  # [n_samples, ps, ps]

        # compute homography
        homograpy = self.compute_homography(
            normal=valid_normal,
            plane_dist=valid_plane_dist,
            camera_ref=cam_ref,
            camera_qry=cam_qry,
        )
        pixels_warp = self.patch_warp(homograpy, valid_pixels)
        normalized_pixels_warp = pixels_warp.clone()
        normalized_pixels_warp[..., 0] = normalized_pixels_warp[..., 0] / (float(W) / 2) - 1.0
        normalized_pixels_warp[..., 1] = normalized_pixels_warp[..., 1] / (float(H) / 2) - 1.0
        gray_patch_qry = F.grid_sample(
            gray_qry.unsqueeze(0), normalized_pixels_warp.reshape(1, -1, 1, 2), align_corners=True
        )
        gray_patch_qry = gray_patch_qry.reshape(self.patch_sample_num, self.patch_size, self.patch_size)
        return gray_patch_ref, gray_patch_qry, valid

    @classmethod
    def patch_offsets(cls, patch_size: int, device: torch.device = torch.device("cpu")):
        """
        :Returns:
        - **offsets**: [patch_size*patch_size, 2]
        """
        half_ps = int((patch_size - 1) / 2)
        offsets = torch.arange(-half_ps, half_ps + 1, device=device).float()
        return torch.stack(torch.meshgrid(offsets, offsets, indexing="xy"), dim=-1).reshape(-1, 2)

    @classmethod
    def compute_homography(cls, normal: torch.Tensor, plane_dist: torch.Tensor, camera_ref: Camera, camera_qry: Camera):
        """
        :Args:
        - **normal**: [N, 3]
        - **plane_dist**: [N,]
        - **camera_ref**: Camera
        - **camera_qry**: Camera

        :Returns:
        - **H**: [N, 3, 3], the homography matrix
        """
        # compute relative pose: ref->query = ref->world->query
        ref2qry = torch.einsum(
            "...nm, ...mk -> ...nk",
            camera_qry.world_to_camera.transpose(-1, -2),
            camera_ref.world_to_camera.transpose(-1, -2).inverse(),
        )
        rel_rot, rel_trans = ref2qry[..., :3, :3], ref2qry[..., :3, -1]  # [3, 3], [3]

        # compute Homography

        H = (
            rel_rot.unsqueeze(0) + torch.einsum("k, n d -> n k d", rel_trans, normal) / plane_dist[..., None, None]
        )  # [n_samples, 3, 3]

        H = torch.einsum("...mn, ...nd -> ...md", camera_qry.get_K()[..., :3, :3].unsqueeze(0), H)
        H = torch.einsum("...mn, ...nd -> ...md", H, camera_ref.get_K()[..., :3, :3].inverse().unsqueeze(0))
        return H

    @classmethod
    def patch_warp(cls, H: torch.Tensor, pixels: torch.Tensor):
        """
        :Args:
        - **H**: [N, 3, 3], the homography matrix
        - **pixels**: [N, M, 2]

        :Returns:
        - **pixels_warp**: [N, M, 2]
        """
        n_samples, n_pixels = pixels.shape[:2]
        assert H.shape[0] == n_samples, "The number of homography matrices must be the same as the number of pixels"
        pixels_warp = torch.einsum(
            "...kd, ...nd -> ...nk", H, torch.cat([pixels, pixels.new_ones((n_samples, n_pixels, 1))], dim=-1)
        )  # [N, M, 3]
        pixels_warp = pixels_warp[..., :2] / (pixels_warp[..., -1:] + 1e-8)
        return pixels_warp

    @classmethod
    def rgb2gray(self, rgb: torch.Tensor):
        """
        :Args:
        - **rgb**: [C, 3, H, W]

        :Returns:
        - **gray**: [C, 1, H, W]
        """
        rgb_coef = torch.tensor([0.2989, 0.5870, 0.1140])
        return torch.einsum("...dhw, d -> ...hw", rgb, rgb_coef.to(rgb)).unsqueeze(1)  # [C, 1, H, W]

    @classmethod
    def lncc(cls, ref, qry):
        # ref_gray: [batch_size, patch_size, patch_size]

        bs, patch_size = qry.shape[:2]

        ref_qry = ref * qry
        ref_qry = ref_qry.view(bs, 1, patch_size, patch_size)
        ref = ref.view(bs, 1, patch_size, patch_size)
        qry = qry.view(bs, 1, patch_size, patch_size)
        ref2 = ref.pow(2)
        qry2 = qry.pow(2)

        # sum over kernel
        # filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
        # padding = patch_size // 2
        # ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
        # qry_sum = F.conv2d(qry, filters, stride=1, padding=padding)[:, :, padding, padding]
        # ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
        # qry2_sum = F.conv2d(qry2, filters, stride=1, padding=padding)[:, :, padding, padding]
        # ref_qry_sum = F.conv2d(ref_qry, filters, stride=1, padding=padding)[:, :, padding, padding]
        ref_sum = ref.sum(dim=[-1, -2])
        qry_sum = qry.sum(dim=[-1, -2])
        ref2_sum = ref2.sum(dim=[-1, -2])
        qry2_sum = qry2.sum(dim=[-1, -2])
        ref_qry_sum = ref_qry.sum(dim=[-1, -2])

        # average over kernel
        ref_avg = ref_sum / (patch_size**2)
        qry_avg = qry_sum / (patch_size**2)

        cross = ref_qry_sum - qry_avg * ref_sum
        ref_var = ref2_sum - ref_avg * ref_sum
        qry_var = qry2_sum - qry_avg * qry_sum

        cc = cross * cross / (ref_var * qry_var + 1e-8)
        ncc = 1 - cc
        ncc = torch.clamp(ncc, 0.0, 2.0)
        # ncc = torch.mean(ncc, dim=1, keepdim=True)
        mask = ncc < 0.9
        return ncc.squeeze(), mask.squeeze()
