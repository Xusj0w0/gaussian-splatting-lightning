from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightning
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from gsplat import spherical_harmonics
from pytorch3d.transforms import quaternion_multiply
from torch_scatter import scatter_mean

from internal.optimizers import Adam
from internal.renderers import RendererOutputInfo, RendererOutputTypes
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule)
from internal.schedulers import ExponentialDecayScheduler
from internal.utils.network_factory import NetworkFactory
# from internal.cameras.cameras import Camera
from myimpl.dataparsers.feature_dataparser import FeatureShapeCamera
from myimpl.models.extended_grid_gaussians import (GridGaussianModel,
                                                   LoDGridGaussianModel,
                                                   ScaffoldGaussianModelMixin)
from myimpl.models.extended_implicit_grid_gaussian import (
    ImplicitGridGaussianModel, ImplicitLoDGridGaussianModel)
from myimpl.renderers.extended_grid_renderer import (
    GridGaussianRenderer, GridGaussianRendererModule, OptimizationConfig)

__all__ = ["GridFeatureGaussianRenderer", "GridFeatureGaussianRendererModule"]


@dataclass
class AdapterOptimizationConfig:
    network_lr_init: float = 2e-3
    network_lr_final: float = 2e-5

    embedding_lr_init: float = 2e-3
    embedding_lr_final: float = 2e-4

    max_steps: int = None


@dataclass
class AdapterConfig:
    network_n_layers: int = 2

    network_hidden_dim: int = 32

    embedding_dim: int = 0

    optimization: AdapterOptimizationConfig = field(default_factory=lambda: AdapterOptimizationConfig())


class Adapter(nn.Module):
    def __init__(self, anchor_feature_fim: int, gt_feat_shape: List[int], num_cameras: int, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.network = NetworkFactory(tcnn=False).get_network(
            n_input_dims=anchor_feature_fim + config.embedding_dim,
            n_output_dims=gt_feat_shape[-1],
            n_layers=config.network_n_layers,
            n_neurons=config.network_hidden_dim,
            activation="ReLU",
            output_activation="None",
        )
        # self.network = nn.Sequential(
        #     nn.Linear(anchor_feature_fim + config.embedding_dim, config.network_hidden_dim),
        #     nn.LayerNorm(config.network_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.network_hidden_dim, gt_feat_shape[-1]),
        # )
        if config.embedding_dim > 0:
            self.embedding = nn.Embedding(
                num_embeddings=num_cameras,
                embedding_dim=config.embedding_dim,
            )

        self.register_buffer("_feature_map_size", torch.tensor(gt_feat_shape[:2]))

    @property
    def feature_map_size(self) -> List[int]:
        self._feature_map_size: torch.Tensor
        if getattr(self, "_feature_map_size", None) is None:
            return None
        return self._feature_map_size.int().tolist()

    def forward(self, x: torch.Tensor, camera_idx: torch.Tensor):
        """
        x: [H, W, C]
        """
        if self.config.embedding_dim > 0:
            embedding: torch.Tensor = self.embedding(camera_idx)
            x = torch.cat([x, embedding.unsqueeze(0).unsqueeze(0).expand(*x.shape[:2], -1)], dim=-1)
        out = self.network(x).permute(2, 0, 1).unsqueeze(0)
        # fmt: off
        out_aligned = F.interpolate(
            out, size=self.feature_map_size, mode="bilinear", align_corners=True
        ).squeeze(0).permute(1, 2, 0) # C' H W --> H W C'
        # fmt: on
        return out_aligned

    def training_setup(self):
        # fmt: off
        net_optimizer = Adam().instantiate(
            [{
                "params": self.network.parameters(),
                "lr": self.config.optimization.network_lr_init,
                "name": "adapter_network",
            }],
            lr=0.0,
        )
        # fmt: on
        net_scheduler = (
            ExponentialDecayScheduler(
                lr_final=self.config.optimization.network_lr_final,
                max_steps=self.config.optimization.max_steps,
            )
            .instantiate()
            .get_scheduler(optimizer=net_optimizer, lr_init=self.config.optimization.network_lr_init)
        )

        optimizers, schedulers = [net_optimizer], [net_scheduler]

        if self.config.embedding_dim > 0:
            # fmt: off
            embedding_optimizer = Adam().instantiate(
                [{
                    "params": self.embedding.parameters(),
                    "lr": self.config.optimization.embedding_lr_init,
                    "name": "adapter_embedding",
                }],
                lr=0.0,
            )
            # fmt: on
            embedding_scheduler = (
                ExponentialDecayScheduler(
                    lr_final=self.config.optimization.embedding_lr_final,
                    max_steps=self.config.optimization.max_steps,
                )
                .instantiate()
                .get_scheduler(optimizer=embedding_optimizer, lr_init=self.config.optimization.embedding_lr_init)
            )
            optimizers.append(embedding_optimizer)
            schedulers.append(embedding_scheduler)

        return optimizers, schedulers


@dataclass
class GridFeatureGaussianRenderer(GridGaussianRenderer):

    feature_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig())

    def instantiate(self, *args, **kwargs):
        return GridFeatureGaussianRendererModule(self)


class GridFeatureGaussianRendererModule(GridGaussianRendererModule):
    config: GridFeatureGaussianRenderer

    def setup(self, stage: str, lightning_module=None, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, lightning_module, *args, **kwargs)

        if stage == "fit":
            anchor_feat_dim = lightning_module.gaussian_model.config.feature_dim
            # hwc shape
            gt_feat_shape = lightning_module.trainer.datamodule.dataparser_outputs.train_set.extra_data[0]["semantic_feature"].shape  # fmt: skip
            num_cameras = len(lightning_module.trainer.datamodule.dataparser_outputs.train_set.cameras)
            self.feature_adapter = Adapter(anchor_feat_dim, gt_feat_shape, num_cameras, self.config.feature_adapter)

    def training_setup(self, module: lightning.LightningModule):
        optimizers, schedulers = super().training_setup(module)

        if self.config.feature_adapter.optimization.max_steps is None:
            self.config.feature_adapter.optimization.max_steps = module.trainer.max_steps
        _optimizers, _schedulers = self.feature_adapter.training_setup()

        optimizers.extend(_optimizers)
        schedulers.extend(_schedulers)
        return optimizers, schedulers

    def rasterize_feature_anchor(
        self,
        viewpoint_camera: FeatureShapeCamera,
        pc: ImplicitGridGaussianModel,
        output_pkg: Dict[str, torch.Tensor],
        scaling_modifier=1.0,
        *args,
        **kwargs,
    ):
        opacities, anchor_mask, primitive_mask = (
            output_pkg["opacities"],
            output_pkg["anchor_mask"],
            output_pkg["primitive_mask"],
        )
        scatter_indices = repeat(
            torch.arange(len(primitive_mask) // pc.n_offsets).to(pc.get_anchors.device), "n -> n o", o=pc.n_offsets
        )
        scatter_indices = scatter_indices.reshape(-1)[primitive_mask]
        opacities = scatter_mean(opacities, scatter_indices, dim=0, dim_size=len(primitive_mask) // pc.n_offsets)

        xyz = pc.get_anchors[anchor_mask].clone().detach()
        feature = pc.get_anchor_features[anchor_mask]
        scales = pc.get_scalings[anchor_mask][..., :3].clone().detach()
        rotations = pc.get_rotations[anchor_mask].clone().detach()
        # opacities = pc.get_opacities[anchor_mask].clone().detach()

        # preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
        preprocessed_camera = viewpoint_camera.preprocess_feature_camera()
        if scaling_modifier != 1.0:
            scales = scales * scaling_modifier

        projections = GSplatV1.project(
            preprocessed_camera,
            xyz,
            scales,
            rotations,
            eps2d=self.config.filter_2d_kernel_size,
            anti_aliased=self.config.anti_aliased,
            radius_clip=self.runtime_options.radius_clip,
            # radius_clip_from=self.runtime_options.radius_clip_from,
            camera_model=self.runtime_options.camera_model,
        )
        radii, means2d, depths, conics, compensations = projections

        # 2. get opacities and then isect encoding
        opacities = opacities.unsqueeze(0).squeeze(-1)  # [1, N]
        if self.config.anti_aliased:
            opacities = opacities * compensations

        isects = self.isect_encode(
            preprocessed_camera,
            projections,
            opacities,
            tile_size=self.config.block_size,
        )

        # 3. rasterization
        means2d = means2d.squeeze(0)
        projection_for_rasterization = radii, means2d, depths, conics, compensations

        render_feature, alpha = GSplatV1.rasterize(
            preprocessed_camera,
            projection_for_rasterization,
            isects,
            opacities,
            colors=feature,
            background=feature.new_zeros((feature.shape[-1],)),
            tile_size=self.config.block_size,
        )
        render_feature = render_feature / (alpha + 1e-8)  # [H, W, C]

        output_pkg.update({"render_feature": render_feature})

        # 4. match gt feature map
        if getattr(self, "feature_adapter", None) is None:
            return output_pkg

        feature_aligned = self.feature_adapter(render_feature, viewpoint_camera.idx)
        output_pkg.update({"render_feature_aligned": feature_aligned})

        return output_pkg

    def rasterize_feature_primitive(
        self,
        viewpoint_camera: FeatureShapeCamera,
        pc: ImplicitGridGaussianModel,
        output_pkg: Dict[str, torch.Tensor],
        scaling_modifier=1.0,
        *args,
        **kwargs,
    ):
        # TODO: feature loss won't decrease

        # preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
        preprocessed_camera = viewpoint_camera.preprocess_feature_camera()

        # reuse properties
        means2d, conics, isects, opacities = (
            output_pkg[k] for k in ["viewspace_points", "conics", "isects", "opacities"]
        )
        opacities = opacities.clone().detach().unsqueeze(0)
        projections = None, means2d.clone().detach(), None, conics.clone().detach(), None

        # get features
        anchor_mask, primitive_mask = output_pkg["anchor_mask"], output_pkg["primitive_mask"]
        features = pc.get_anchor_features[anchor_mask]
        # features = repeat(features, "n m -> (n o) m", o=pc.n_offsets)
        # features = features.expand(features.shape[0]*pc.n_offsets, -1)
        features = features.unsqueeze(0).expand(pc.n_offsets, -1, -1).permute(1, 0, 2).reshape(-1, features.shape[-1])
        features = features[primitive_mask]

        render_feature, _ = GSplatV1.rasterize(
            preprocessed_camera,
            projections,
            isects,
            opacities,
            features,
            features.new_zeros((features.shape[-1],)),
            tile_size=self.config.block_size,
            absgrad=False,
        )

        output_pkg.update({"render_feature": render_feature})

        if getattr(self, "feature_adapter", None) is None:
            return output_pkg

        feature_aligned = self.feature_adapter(render_feature, viewpoint_camera.idx)  # H W C' --> 1 C' H W
        output_pkg.update({"render_feature_aligned": feature_aligned})

        return output_pkg

    def forward(
        self,
        viewpoint_camera: FeatureShapeCamera,
        pc: ImplicitGridGaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        render_types: list = None,
        **kwargs,
    ):
        if render_types is None:
            known_types = None
        else:
            known_types = list(filter(lambda x: x in self.RENDER_TYPE_BITS, render_types))

        output_pkg = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, known_types, **kwargs)

        # render features
        output_pkg = self.rasterize_feature_anchor(viewpoint_camera, pc, output_pkg, scaling_modifier, **kwargs)
        # output_pkg = self.rasterize_feature_primitive(viewpoint_camera, pc, output_pkg, scaling_modifier, **kwargs)

        return output_pkg

    def get_rgbs_from_SHs(
        self,
        camera: FeatureShapeCamera,
        xyz: torch.Tensor,
        colors: torch.Tensor,
        visibility_filter: torch.Tensor,
        activate_sh_degree: int,
    ):
        viewdirs = xyz.detach() - camera.camera_center
        return spherical_harmonics(activate_sh_degree, viewdirs, colors, visibility_filter)

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        super().setup_web_viewer_tabs(viewer, server, tabs)

        # if isinstance(viewer.gaussian_model, LoDGridGaussianModel):
        if (
            getattr(viewer.gaussian_model, "get_levels", None) is not None
            and viewer.gaussian_model.get_levels.shape[0] > 0
        ):
            with tabs.add_tab("Octree"):
                self._lod_options = ViewerOptions(viewer, server)

    @property
    def feature_map_size(self) -> List[int]:
        self._feature_map_size: torch.Tensor
        if getattr(self, "_feature_map_size", None) is None:
            return None
        return self._feature_map_size.int().tolist()

    def get_available_outputs(self):
        available_outputs = super().get_available_outputs()
        extra_outputs = {
            "render_feature": {
                "render_feature": RendererOutputInfo("render_feature", RendererOutputTypes.FEATURE_MAP),
            }
        }
        available_outputs.update(extra_outputs)
        return available_outputs


from viser import ViserServer

from internal.viewer.viewer import Viewer


class ViewerOptions:
    def __init__(self, viewer: Viewer, server: ViserServer):
        self.viewer = viewer
        self.server = server

        self.activate_level_slider = server.gui.add_slider(
            label="Activate LoD Level",
            min=0,
            step=1,
            max=viewer.gaussian_model.max_level,
            initial_value=viewer.gaussian_model.max_level,
        )

        @self.activate_level_slider.on_update
        def _(_):
            viewer.viewer_renderer.gaussian_model.activate_level = self.activate_level_slider.value
            viewer.rerender_for_all_client()
