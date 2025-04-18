import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser(description="Extract SVD decomposition")
    parser.add_argument("feature_map_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_svd_decomposition(feature_map_path, generator=None):
    feature_map = np.load(feature_map_path)
    feature_map = torch.from_numpy(feature_map).float().cuda()
    feature_map = F.normalize(feature_map, dim=-1)
    flattened_feature = feature_map.reshape(-1, feature_map.shape[-1])

    randint = torch.randint(0, flattened_feature.shape[0], [1 << 18], generator=generator)
    X = flattened_feature[randint, :]
    n = X.shape[0]
    mean = torch.mean(X, dim=0)
    X = X - mean  # [N, d]
    S = (1 / (n - 1)) * torch.matmul(X.T, X).float()
    D_inv = torch.diag(1.0 / torch.sqrt(torch.diag(S)))
    normalized_S = D_inv @ S @ D_inv

    eigenvalues, eigenvectors = torch.linalg.eig(normalized_S)
    idx = torch.argsort(-eigenvalues.real)
    eigenvectors = eigenvectors.real[:, idx]
    eigenvalues = eigenvalues.real[idx]
    return eigenvalues, eigenvectors


if __name__ == "__main__":
    args = make_parser().parse_args()
    feature_map_dir = args.feature_map_dir
    output_dir = args.output

    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    if output_dir is None:
        output_dir = osp.join(osp.dirname(feature_map_dir), f"{osp.basename(feature_map_dir)}_decomp")
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for feature_map_file in tqdm(os.listdir(feature_map_dir)):
        feature_map_path = osp.join(feature_map_dir, feature_map_file)
        eigenvalues, eigenvectors = get_svd_decomposition(feature_map_path, generator)
        torch.save(
            {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors},
            osp.join(output_dir, f"{feature_map_file.rsplit('.', 1)[0]}.pt"),
        )
