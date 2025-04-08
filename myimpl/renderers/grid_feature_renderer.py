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

from internal.optimizers import Adam
from internal.renderers.gsplat_v1_renderer import GSplatV1, GSplatV1Renderer, GSplatV1RendererModule
from internal.schedulers import ExponentialDecayScheduler
from internal.utils.network_factory import NetworkFactory

# from internal.cameras.cameras import Camera
from myimpl.dataparsers.feature_dataparser import FeatureShapeCamera
from myimpl.models.extended_grid_gaussians import GridGaussianModel, LoDGridGaussianModel, ScaffoldGaussianModelMixin
from myimpl.models.extended_implicit_grid_gaussian import ImplicitGridGaussianModel, ImplicitLoDGridGaussianModel
from myimpl.renderers.extended_grid_renderer import GridGaussianRenderer, GridGaussianRendererModule
from myimpl.renderers.extended_grid_renderer import OptimizationConfig as _OptimizationConfig

__all__ = ["GridFeatureGaussianRenderer", "GridFeatureGaussianRendererModule"]


@dataclass
class AppearanceModelConfig:
    n_appearances: int = -1


@dataclass
class OptimizationConfig(_OptimizationConfig):
    adapter_lr_init: float = 0.005

    adapter_lr_final: float = 0.0005

    adapter_n_layers: int = 2

    adapter_hidden_dim: int = 64


@dataclass
class GridFeatureGaussianRenderer(GridGaussianRenderer):
    anti_aliased: bool = False

    model: AppearanceModelConfig = field(default_factory=lambda: AppearanceModelConfig())

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

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
            self.feature_adapter = NetworkFactory(tcnn=False).get_network(
                n_input_dims=anchor_feat_dim,
                n_output_dims=gt_feat_shape[-1],
                n_layers=self.config.optimization.adapter_n_layers,
                n_neurons=self.config.optimization.adapter_hidden_dim,
                activation="ReLU",
                output_activation="None",
            )
            self.register_buffer("_feature_map_size", torch.tensor(gt_feat_shape[:2]))

    def training_setup(self, module: lightning.LightningModule):
        optimizers, schedulers = super().training_setup(module)
        # fmt: off
        optimizer = Adam().instantiate(
            [{
                "params": self.feature_adapter.parameters(),
                "lr": self.config.optimization.adapter_lr_init,
                "name": "feature_adapter",
            }],
            lr=0.0,
        )
        # fmt: on
        scheduler = (
            ExponentialDecayScheduler(
                lr_final=self.config.optimization.adapter_lr_final,
                max_steps=self.config.optimization.max_steps,
            )
            .instantiate()
            .get_scheduler(optimizer=optimizer, lr_init=self.config.optimization.adapter_lr_init)
        )
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        return optimizer, schedulers

    def rasterize_feature_lowres(
        self,
        viewpoint_camera: FeatureShapeCamera,
        pc: ImplicitGridGaussianModel,
        anchor_mask: torch.Tensor,
        scaling_modifier=1.0,
        *args,
        **kwargs,
    ):
        xyz = pc.get_anchors[anchor_mask].clone().detach()
        feature = pc.get_anchor_features[anchor_mask]
        scales = pc.get_scalings[anchor_mask][..., :3].clone().detach()
        rotations = pc.get_rotations[anchor_mask].clone().detach()
        opacities = pc.get_opacities[anchor_mask].clone().detach()

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
        render_feature = render_feature / (alpha + 1e-8)

        return {"render_feature": render_feature}

    def rasterize_feature(
        self,
        viewpoint_camera: FeatureShapeCamera,
        pc: ImplicitGridGaussianModel,
        anchor_mask: torch.Tensor,
        scaling_modifier=1.0,
        *args,
        **kwargs,
    ):
        xyz = pc.get_anchors[anchor_mask]
        feature = pc.get_anchor_features[anchor_mask]
        scales = pc.get_scalings[anchor_mask][..., :3]
        rotations = pc.get_rotations[anchor_mask]
        opacities = pc.get_opacities[anchor_mask]

        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
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

        output_pkg = {"render_feature": render_feature}

        # 4. match gt feature map
        if getattr(self, "feature_adapter", None) is None:
            return output_pkg

        feature_aligned = self.feature_adapter(render_feature).permute(2, 0, 1).unsqueeze(0)  # H W C' --> 1 C' H W
        # fmt: off
        feature_aligned = F.interpolate(
            feature_aligned, size=self.feature_map_size, mode="bilinear", align_corners=True
        ).squeeze(0).permute(1, 2, 0) # C' H W --> H W C'
        # fmt: on
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
        output_pkg = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, render_types, **kwargs)

        # render features
        anchor_mask = output_pkg["anchor_mask"]
        feature_output_pkg = self.rasterize_feature(viewpoint_camera, pc, anchor_mask, scaling_modifier, **kwargs)
        output_pkg.update(feature_output_pkg)

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
