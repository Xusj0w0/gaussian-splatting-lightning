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
from myimpl.models.grid_gaussians import (GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)
from myimpl.models.implicit_grid_gaussian import (ImplicitGridGaussianModel,
                                                  ImplicitLoDGridGaussianModel)
from myimpl.renderers.grid_renderer import (GridGaussianRenderer,
                                            GridGaussianRendererModule,
                                            OptimizationConfig)

__all__ = ["GridFeatureGaussianRenderer", "GridFeatureGaussianRendererModule"]


@dataclass
class GridFeatureGaussianRenderer(GridGaussianRenderer):
    render_feature_size: int = 256
    """short side of the feature map"""

    def instantiate(self, *args, **kwargs):
        return GridFeatureGaussianRendererModule(self)


class GridFeatureGaussianRendererModule(GridGaussianRendererModule):
    config: GridFeatureGaussianRenderer

    def setup(self, stage: str, lightning_module=None, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, lightning_module, *args, **kwargs)

    def training_setup(self, module: lightning.LightningModule):
        optimizers, schedulers = super().training_setup(module)

        return optimizers, schedulers

    def preprocess_feature_camera(self, viewpoint_camera: FeatureShapeCamera):
        if viewpoint_camera.width > viewpoint_camera.height:
            h = int(self.config.render_feature_size)
            w = int(round(self.config.render_feature_size * float(viewpoint_camera.width / viewpoint_camera.height)))
        else:
            w = int(self.config.render_feature_size)
            h = int(round(self.config.render_feature_size * float(viewpoint_camera.width / viewpoint_camera.height)))
        scale_x, scale_y = float(w) / viewpoint_camera.width.item(), float(h) / viewpoint_camera.height.item()

        viewmats = viewpoint_camera.world_to_camera.T.unsqueeze(0)
        # fmt: off
        Ks = torch.tensor([[
            [viewpoint_camera.fx * scale_x, 0, viewpoint_camera.cx * scale_x],
            [0.0, viewpoint_camera.fy * scale_y, viewpoint_camera.cy * scale_y],
            [0.0, 0.0, 1.0]
        ]], dtype=torch.float, device=viewpoint_camera.R.device)
        # fmt: on

        return viewmats, Ks, (w, h)

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
        features = pc.get_anchor_features[anchor_mask]
        scales = pc.get_scalings[anchor_mask][..., :3].clone().detach()
        rotations = pc.get_rotations[anchor_mask].clone().detach()
        # opacities = pc.get_opacities[anchor_mask].clone().detach()

        # preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
        preprocessed_camera = self.preprocess_feature_camera(viewpoint_camera)
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
            colors=features,
            background=features.new_zeros((features.shape[-1],)),
            tile_size=self.config.block_size,
        )
        # render_feature = render_feature / (alpha + 1e-8)  # [H, W, C]

        output_pkg.update({"render_feature": render_feature})

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

        # reuse implicit properties
        xyz, scales, rotations, opacities = (
            output_pkg[i].clone().detach() for i in ["xyz", "scales", "rotations", "opacities"]
        )

        anchor_mask, primitive_mask = output_pkg["anchor_mask"], output_pkg["primitive_mask"]
        features = pc.get_anchor_features[anchor_mask]
        features = features.unsqueeze(0).expand(pc.n_offsets, -1, -1).permute(1, 0, 2).reshape(-1, features.shape[-1])
        features = features[primitive_mask]

        # preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
        preprocessed_camera = self.preprocess_feature_camera(viewpoint_camera)
        if scaling_modifier != 1.0:
            scales = scales * scaling_modifier

        # former projection results are calculated with original camera params
        # need to re-calculate projection
        projections = GSplatV1.project(
            preprocessed_camera,
            xyz,
            scales,
            rotations,
            eps2d=self.config.filter_2d_kernel_size,
            anti_aliased=self.config.anti_aliased,
            radius_clip=self.runtime_options.radius_clip,
            camera_model=self.runtime_options.camera_model,
        )
        radii, means2d, depths, conics, compensations = projections

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
            colors=features,
            background=features.new_zeros((features.shape[-1],)),
            tile_size=self.config.block_size,
        )

        output_pkg.update({"render_feature": render_feature})

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
        # output_pkg = self.rasterize_feature_anchor(viewpoint_camera, pc, output_pkg, scaling_modifier, **kwargs)
        output_pkg = self.rasterize_feature_primitive(viewpoint_camera, pc, output_pkg, scaling_modifier, **kwargs)

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
