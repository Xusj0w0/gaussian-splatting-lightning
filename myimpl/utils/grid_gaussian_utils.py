import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


def init_weight(module: nn.Module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        if module.bias is not None:
            nn.init.trunc_normal_(module.bias, 0, 0.01, a=-0.04, b=0.04)


def knn(x, K):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class GridGaussianUtils:
    chunk_size_max = 1 << 22

    @staticmethod
    def max_power_of_2(n: int) -> int:
        assert n >= 1
        power, next = 1, 2
        while next <= n:
            power = next
            next *= 2
        return power

    @staticmethod
    def get_levels_by_distances(
        points: torch.Tensor,
        camera_infos: torch.Tensor,
        dist_ratio: float = 0.001,
        fork: int = 2,
        use_chunk=True,
    ):
        num_points = len(points)
        points = points.to(camera_infos).unsqueeze(0)

        if use_chunk:
            # chunk cameras, since quantile is applied on points
            chunk_size = GridGaussianUtils.max_power_of_2(GridGaussianUtils.chunk_size_max // num_points)
            dists_min = camera_infos.new_zeros((camera_infos.shape[0],))
            dists_max = camera_infos.new_zeros((camera_infos.shape[0],))
            for st in range(0, camera_infos.shape[0], chunk_size):
                ed = min(st + chunk_size, camera_infos.shape[0])
                _camera_infos = camera_infos[st:ed]
                ds = (
                    torch.sqrt(torch.sum((points - _camera_infos[:, :3].unsqueeze(1)) ** 2, dim=-1))
                    * _camera_infos[:, -1:]
                )
                dists_min[st:ed].copy_(torch.quantile(ds, dist_ratio, dim=-1))
                dists_max[st:ed].copy_(torch.quantile(ds, 1 - dist_ratio, dim=-1))
        else:
            ds = torch.sqrt(torch.sum((points - camera_infos[:, :3].unsqueeze(1)) ** 2, dim=-1)) * camera_infos[:, -1:]
            dists_min = torch.quantile(ds, dist_ratio, dim=-1)
            dists_max = torch.quantile(ds, 1 - dist_ratio, dim=-1)

        dist_min = torch.quantile(dists_min, dist_ratio)
        dist_max = torch.quantile(dists_max, 1 - dist_ratio)

        max_level = torch.round(torch.log2(dist_max / dist_min) / math.log2(float(fork))).int() + 1
        return dist_max, max_level

    @staticmethod
    def get_coarse_intervals(num_level: int, coarse_iter: int, coarse_factor: float):
        coarse_intervals = []
        if num_level > 0:
            q = 1.0 / coarse_factor
            a1 = coarse_iter * (1 - q) / (1 - q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q**i + temp_interval
                temp_interval = interval
                coarse_intervals.append(interval)
        return coarse_intervals

    def build_grid(points: torch.Tensor, default_voxel_size: float):
        grid_origin = points.mean(dim=0)
        voxel_size = torch.tensor(default_voxel_size)
        if voxel_size <= 0:
            from simple_knn._C import distCUDA2

            _points = points.clone().cuda()
            _dist = distCUDA2(_points)
            median_dist, _ = torch.kthvalue(_dist, int(len(_points) * 0.5))
            voxel_size = median_dist.to(points.device)
        return voxel_size, grid_origin

    @staticmethod
    def build_multi_level_grid(
        points: torch.Tensor,
        extend_ratio: float,
        base_layer: int,
        fork: int = 2,
        default_voxel_size: Optional[float] = None,
        max_level: Optional[int] = None,
    ):
        box_min, box_max = torch.min(points, dim=0).values, torch.max(points, dim=0).values
        extend_min = box_min - (box_max - box_min) * extend_ratio
        extend_max = box_max + (box_max - box_min) * extend_ratio
        box_d = torch.max(extend_max - extend_min)

        if base_layer < 0:
            assert default_voxel_size is not None and max_level is not None
            base_layer = torch.round(torch.log2(box_d / default_voxel_size)).int().item() - (max_level // 2) + 1
        voxel_size = box_d / (float(fork) ** base_layer)
        return voxel_size, points.mean(dim=0)

    @staticmethod
    def map_to_int_level(pred_level: torch.Tensor, cur_level: int, dist2level: str = "floor"):
        if dist2level == "floor":
            return torch.floor(pred_level).int().clamp(min=0, max=cur_level), None
        elif dist2level == "round":
            return torch.round(pred_level).int().clamp(min=0, max=cur_level), None
        elif dist2level == "ceil":
            return torch.ceil(pred_level).int().clamp(min=0, max=cur_level), None
        elif dist2level == "progressive":
            eps = 1e-4
            pred_level = torch.clamp(pred_level + 1.0, min=1.0 - eps, max=cur_level - eps)
            int_level = torch.floor(pred_level).int()
            prog_ratio = torch.frac(pred_level)
            return int_level, prog_ratio
        else:
            raise ValueError(f"Unknown dist2level: {dist2level}")

    @staticmethod
    def predict_level(dists: torch.Tensor, standard_dist: float, fork: int = 2):
        return torch.log2(standard_dist / dists) / math.log2(fork)

    @staticmethod
    def weed_out_mask_by_level(
        anchors: torch.Tensor,
        levels: torch.Tensor,
        vis_thresh: float,
        cam_infos: torch.Tensor,
        predict_level_fn: Callable,
        int_level_fn: Callable,
        use_chunk: bool = True,
    ) -> torch.Tensor:
        if use_chunk:
            chunk_size = GridGaussianUtils.max_power_of_2(GridGaussianUtils.chunk_size_max // cam_infos.shape[0])
            count = anchors.new_zeros((anchors.shape[0],))
            for st in range(0, anchors.shape[0], chunk_size):
                ed = min(st + chunk_size, anchors.shape[0])
                _anchor, _level = anchors[st:ed].reshape(-1, 1, 3), levels[st:ed].reshape(-1, 1)
                dists = torch.sqrt(torch.sum((_anchor - cam_infos[:, :3].reshape(-1, 3)) ** 2, dim=-1))
                pred_level = predict_level_fn(dists)
                int_level = int_level_fn(pred_level)
                # if anchor level is lower than level pred by camera
                # then the anchor is coarse and visible
                count[st:ed].copy_((_level <= int_level).sum(dim=1).float())
            count /= len(cam_infos)
        else:
            dists = torch.sqrt(torch.sum((_anchor - cam_infos[:, :3].reshape(-1, 3)) ** 2, dim=-1))
            pred_level = predict_level_fn(dists)
            int_level = int_level_fn(pred_level)
            count = (levels.reshape(-1, 1) <= int_level).sum(dim=-1).float()

        mask = count > vis_thresh
        return mask

    @staticmethod
    def point_to_grid(
        points: torch.Tensor, voxel_size: float, grid_origin: torch.Tensor, padding: float = 0.0
    ) -> torch.Tensor:
        return torch.round((points - grid_origin.to(points)) / voxel_size + padding).int()

    @staticmethod
    def grid_to_point(
        grid: torch.Tensor, voxel_size: float, grid_origin: torch.Tensor, padding: float = 0.0
    ) -> torch.Tensor:
        return (grid.float() - padding) * voxel_size + grid_origin.to(grid.device)

    @staticmethod
    def voxelize(points: torch.Tensor, voxel_size: float, xyz2grid: Callable, grid2xyz: Callable) -> torch.Tensor:
        return grid2xyz(torch.unique(xyz2grid(points, voxel_size), dim=0), voxel_size)

    @staticmethod
    def multi_level_voxelize(
        points: torch.Tensor, voxel_size: float, max_level: int, xyz2grid: Callable, grid2xyz: Callable, fork: int = 2
    ) -> Tuple[torch.Tensor]:
        positions = points.new_empty((0, 3))
        levels = torch.empty(0).int()
        for cur_level in range(max_level):
            cur_size = voxel_size / (float(fork) ** cur_level)

            _positions = GridGaussianUtils.voxelize(points, cur_size, xyz2grid, grid2xyz)
            _levels = levels.new_ones((_positions.shape[0],)) * cur_level

            positions = torch.cat((positions, _positions), dim=0)
            levels = torch.cat((levels, _levels), dim=0)
        return positions, levels
