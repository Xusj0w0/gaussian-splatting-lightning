import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


def knn(x, K):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def map_to_int_level_factory(dist2level: str = "round"):
    def map_to_int_level(pred_level, cur_level, levels=None):
        if dist2level == "floor":
            return (torch.floor(pred_level).int().clamp(min=0, max=cur_level),)
        elif dist2level == "round":
            return (torch.round(pred_level).int().clamp(min=0, max=cur_level),)
        elif dist2level == "progressive":
            assert levels is not None
            pred_level = torch.clamp(pred_level + 1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            transition_mask = levels.squeeze(1) == int_level
            return (int_level, prog_ratio, transition_mask)
        else:
            raise ValueError(f"Unknown dist2level: {dist2level}")

    return map_to_int_level


def xyz_grid_mapping_factory(init_pos, padding):
    def xyz_to_grid(xyz, voxel_size):
        return torch.round((xyz - init_pos.to(xyz)) / voxel_size + padding).int()

    def grid_to_xyz(grid, voxel_size):
        return (grid - padding) * voxel_size + init_pos.to(grid)

    return xyz_to_grid, grid_to_xyz
