import os
import os.path as osp
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightning
import torch
from appearance_network import AppearanceNetwork
from torch import nn

from internal.cameras import Camera
from internal.models.gaussian import GaussianModel
from internal.renderers.gsplat_v1_renderer import (GSplatV1Renderer,
                                                   GSplatV1RendererModule)


@dataclass
class ModelConfig:
    # num_input_channels: int = -1
    num_output_channels: int = 3
    n_appearances: int = -1
    n_appearance_embedding_dims: int = 32


@dataclass
class OptimizationConfig:
    gamma_eps: float = 1e-6
    embedding_lr_init: float = 2e-3
    embedding_lr_final_factor: float = 0.1
    lr_init: float = 1e-3
    lr_final_factor: float = 0.1
    eps: float = 1e-15
    max_steps: int = 30_000
    warm_up: int = 4000


class AppearanceModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._setup()

    def _setup(self):
        self.embedding = nn.Embedding(self.config.n_appearances, self.config.n_appearance_embedding_dims)
        self.appearance_model = AppearanceNetwork(
            self.config.n_appearance_embedding_dims + 3, self.config.num_output_channels
        )

    def forward(self, rendered: torch.Tensor, appearance_id: torch.Tensor):
        appearance_embedding = self.embedding(appearance_id)
        h, w = list(rendered.shape[-2:])
        crop_image_down = torch.nn.functional.interpolate(
            rendered[None], size=(h // 32, w // 32), mode="bilinear", align_corners=True
        )[0]
        net_input = torch.cat(
            [crop_image_down, appearance_embedding[None, None].repeat(h // 32, w // 32, 1).permute(2, 0, 1)], dim=0
        )[None]

        return self.appearance_model(net_input, h, w).squeeze() * rendered


@dataclass
class GSplatAppearanceRenderer(GSplatV1Renderer):
    separate_sh: bool = True

    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return GSplatAppearanceRendererModule(self)


class GSplatAppearanceRendererModule(GSplatV1RendererModule):
    def setup(self, stage: str, lightning_module=None, *args: Any, **kwargs: Any) -> Any:
        if lightning_module is not None:
            if self.config.model.n_appearances <= 0:
                max_input_id = 0
                appearance_group_ids = lightning_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
                if appearance_group_ids is not None:
                    for i in appearance_group_ids.values():
                        if i[0] > max_input_id:
                            max_input_id = i[0]
                n_appearances = max_input_id + 1
                self.config.model.n_appearances = n_appearances

        self._setup_model()
        print(self.model)

    def _setup_model(self, device=None):
        self.model = AppearanceModel(self.config.model)
        if device is not None:
            self.model.to(device=device)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.config.model.n_appearances = state_dict["model.embedding.weight"].shape[0]
        self._setup_model(device=state_dict["model.embedding.weight"].device)
        return super().load_state_dict(state_dict, strict)

    def training_setup(self, module: lightning.LightningModule):
        embedding_optimizer, _ = self._create_optimizer_and_scheduler(
            self.model.embedding.parameters(),
            "embedding",
            lr_init=self.config.optimization.embedding_lr_init,
            lr_final_factor=self.config.optimization.embedding_lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.warm_up,
        )
        appearance_model_optimizer, _ = self._create_optimizer_and_scheduler(
            self.model.appearance_model.parameters(),
            "appearance_model",
            lr_init=self.config.optimization.lr_init,
            lr_final_factor=self.config.optimization.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.warm_up,
        )
        return [embedding_optimizer, appearance_model_optimizer], None

    def training_forward(
        self,
        step: int,
        module: lightning.LightningModule,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        **kwargs,
    ):
        output_dict = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, **kwargs)
        output_dict.update({"appearance_augmented": self.model(output_dict["render"], viewpoint_camera.appearance_id)})
        return output_dict

    @staticmethod
    def _create_optimizer_and_scheduler(
        params, name, lr_init, lr_final_factor, max_steps, eps, warm_up
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            params=[{"params": list(params), "name": name}],
            lr=lr_init,
            eps=eps,
        )
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda iter: lr_final_factor ** min(max(iter - warm_up, 0) / max_steps, 1),
        #     verbose=False,
        # )

        return optimizer, None
