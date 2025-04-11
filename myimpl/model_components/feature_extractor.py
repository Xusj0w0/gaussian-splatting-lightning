import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter.*",
    category=UserWarning,
)
from abc import ABC, abstractmethod
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class FeatureExtractorType:
    SAM: int = 1
    "segment anything"

    SAM2: int = 2
    "segment anything 2"

    DINOv2: int = 3
    "dino v2"

    DA2: int = 4
    "depth anything v2"


class ExtractorBase(ABC):
    @abstractmethod
    def preprocess_image(self, img: torch.Tensor):
        """
        img: [C, H, W]
        """
        pass

    @abstractmethod
    def get_embedding_shape(self):
        pass


class SAM2Extractor(ExtractorBase, nn.Module):
    def __init__(
        self,
        arch: str = "sam2.1_hiera_l",
        feat_dim: int = Literal[32, 64, 256],
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        assert feat_dim in [32, 64, 256], "feat_dim should be in [32, 64, 256]"
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.feat_dim = feat_dim
        self.device = device

        model_type = arch.rsplit("_", 1)[-1]
        if model_type == "l":
            model_type = "large"
        elif model_type == "b+":
            model_type = "base_plus"
        elif model_type == "s":
            model_type = "small"
        elif model_type == "t":
            model_type = "tiny"

        cfg = f"configs/{arch.split('_', 1)[0]}/{arch}.yaml"
        ckpt = f"checkpoints/sam2/{arch.rsplit('_', 1)[0]}_{model_type}.pt"
        sam = build_sam2(cfg, ckpt, device=device)
        self._predictor: SAM2ImagePredictor = SAM2ImagePredictor(sam)

    def preprocess_image(self, img: torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        min_size = min(*img.shape[:2])
        img = cv2.resize(img, (min_size, min_size), interpolation=cv2.INTER_LINEAR)
        return img

    def get_embedding_shape(self):
        if getattr(self, "_embedding_shape", None) is None:
            img = torch.rand(3, 224, 224).float()
            image_embedding = self(img)
            self._embedding_shape = image_embedding.shape
        return self._embedding_shape

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        img = self.preprocess_image(img)
        self._predictor.set_image(img)

        if self.feat_dim == 256:
            image_embedding = self._predictor._features["image_embed"]  # [C, H, W]
        elif self.feat_dim == 64:
            image_embedding = self._predictor._features["high_res_feats"][1]
        elif self.feat_dim == 32:
            image_embedding = self._predictor._features["high_res_feats"][0]
        else:
            raise ValueError(f"feat_dim {self.feat_dim} not supported")
        image_embedding = image_embedding.squeeze().permute(1, 2, 0)
        return image_embedding


class DINOv2Extractor(ExtractorBase, nn.Module):
    def __init__(
        self,
        arch: str = "dinov2_vitl14_reg",
        img_size: int = 224,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.img_size = img_size
        self.device = device
        try:
            sys.path.insert(0, "external/dinov2")
            import external.dinov2.hubconf as dino_hubconf
            from external.dinov2.dinov2.models.vision_transformer import \
                DinoVisionTransformer

            entry = dino_hubconf.__dict__[arch]
            self._predictor: DinoVisionTransformer = entry().to(device).eval()
        except:
            print("Error importing DINOv2. Please ensure the path is correct.")
            raise

        assert (
            img_size % self._predictor == 0
        ), f"img_size {img_size} should be divisible by {self._predictor.patch_size}"

    def preprocess_image(self, img: torch.Tensor):
        img = TF.resize(img, (self.img_size, self.img_size), interpolation=InterpolationMode.BILINEAR)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img.unsqueeze(0).to(self.device)

    def get_embedding_shape(self):
        if getattr(self, "_embedding_shape", None) is None:
            img = torch.rand(3, 224, 224).float()
            image_embedding = self(img)
            self._embedding_shape = image_embedding.shape
        return self._embedding_shape

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        img = self.perprocess_image(img)

        feature_size = (torch.tensor(img.shape[-2:]) / self._predictor.patch_size).int().tolist()
        image_embedding = self._predictor.forward_features(img)["x_norm_patchtokens"].squeeze(0).permute(1, 0)
        image_embedding = image_embedding.reshape(-1, *feature_size)
        return image_embedding.permute(1, 2, 0)
