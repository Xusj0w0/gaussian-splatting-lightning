import argparse
import glob
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .simple_autoencoder import Autoencoder

# from internal.dataset import CacheDataLoader


class FeatureDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, index):
        file_path = self.files[index]
        features = np.load(file_path)
        return torch.tensor(index), torch.from_numpy(features)[0].permute(2, 0, 1).float()

    def __len__(self):
        return len(self.files)


class AutoencoderTrainer:
    def __init__(
        self,
        max_epoches: int = 100,
        batch_size: int = 64,
        dataset_path: str = None,
        output_path: str = None,
        lr: float = 1e-4,
        input_dim: int = 384,
        encoder_dims: List[int] = [256, 256, 128, 128, 64],
        decoder_dims: List[int] = [64, 128, 128, 256, 256],
    ):
        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.lr = lr

        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        self.autoencoder = Autoencoder(input_dim, encoder_dims, decoder_dims)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autoencoder.to(self.device)

        os.makedirs(osp.join(self.output_path, "logs"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=osp.join(self.output_path, "logs"))

        self.feature_files = glob.glob(osp.join(dataset_path, "*.npy"))
        self.train_loader = DataLoader(
            FeatureDataset([f for i, f in enumerate(self.feature_files) if i % 10 != 0]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            FeatureDataset([f for i, f in enumerate(self.feature_files) if i % 10 == 0]),
            batch_size=self.batch_size * 4,
            shuffle=False,
        )

    def train_one_epoch(self):
        self.autoencoder.train()
        for idx, (_, feature) in enumerate(self.train_loader):
            feature = feature.to(self.device)
            self.optimizer.zero_grad()
            output = self.autoencoder(feature)
            loss = F.mse_loss(output, feature)
            loss.backward()
            self.optimizer.step()

            if idx % 10 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.epoch * len(self.train_loader) + idx)

    def train(self):
        for self.epoch in range(self.max_epoches):
            self.train_one_epoch()

            if (self.epoch + 1) % 10 == 0:
                self.evaluate()
                self.save_model()

    def save_model(self):
        os.makedirs(osp.join(self.output_path, "checkpoints"), exist_ok=True)
        torch.save(
            self.autoencoder.state_dict(), osp.join(self.output_path, "checkpoints", f"epoch_{self.epoch+1}.pth")
        )

    def evaluate(self):
        self.autoencoder.eval()
        losses = []
        for _, feature in self.val_loader:
            feature = feature.to(self.device)
            output = self.autoencoder(feature)
            loss = torch.mean((output - feature) ** 2, dim=[1, 2, 3])
            losses.append(loss)
        losses = torch.cat(losses, dim=0)
        mean_loss = torch.mean(losses)
        self.writer.add_scalar("val/loss", mean_loss.item(), self.epoch)

    def reduce(self):
        self.autoencoder.eval()
        loader = DataLoader(
            FeatureDataset(self.feature_files),
            batch_size=self.batch_size * 4,
            shuffle=False,
        )
        os.makedirs(osp.join(self.output_path, "reduced_features"), exist_ok=True)
        with torch.no_grad():
            for ids, feature in loader:
                feature = feature.to(self.device)
                feature_reduced = self.autoencoder.encode(feature)
                results = feature_reduced.permute(0, 2, 3, 1).cpu().numpy()
                for i, result in zip(ids, results):
                    np.save(
                        osp.join(
                            self.output_path,
                            "reduced_features",
                            f"{osp.basename(self.feature_files[i].rsplit('.', 1)[0])}.npy",
                        ),
                        result,
                    )
