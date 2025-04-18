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

from internal.cameras.cameras import Camera
from internal.optimizers import Adam
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule)
from internal.schedulers import ExponentialDecayScheduler
from myimpl.models.grid_gaussians import (GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)
from myimpl.models.implicit_grid_gaussian import (ImplicitGridGaussianModel,
                                                  ImplicitLoDGridGaussianModel)

__all__ = ["GridGaussianRenderer", "GridGaussianRendererModule"]


@dataclass
class AppearanceModelConfig:
    n_appearances: int = -1


@dataclass
class OptimizationConfig:
    appearance_embedding_lr_init: float = 0.05

    appearance_embedding_lr_final: float = 0.0005

    max_steps: int = None

    eps: float = 1e-15


@dataclass
class GridGaussianRenderer(GSplatV1Renderer):
    anti_aliased: bool = False

    appearance_model: AppearanceModelConfig = field(default_factory=lambda: AppearanceModelConfig())

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return GridGaussianRendererModule(self)


class GridGaussianRendererModule(GSplatV1RendererModule):
    config: GridGaussianRenderer

    def setup(self, stage: str, lightning_module=None, *args: Any, **kwargs: Any) -> Any:
        if self.config.optimization.max_steps is None:
            self.config.optimization.max_steps = lightning_module.trainer.max_steps

        self.n_appearance_embedding_dims = 0
        if lightning_module is not None:
            if self.config.appearance_model.n_appearances <= 0:
                max_input_id = 0
                appearance_group_ids = lightning_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
                if appearance_group_ids is not None:
                    for i in appearance_group_ids.values():
                        if i[0] > max_input_id:
                            max_input_id = i[0]
                n_appearances = max_input_id + 1
                self.config.appearance_model.n_appearances = n_appearances
            self.n_appearance_embedding_dims = lightning_module.gaussian_model.config.n_appearance_embedding_dims

        if self.n_appearance_embedding_dims > 0:
            self.appearance_embedding = nn.Embedding(
                num_embeddings=self.config.appearance_model.n_appearances,
                embedding_dim=self.n_appearance_embedding_dims,
            )

    def training_setup(self, module: lightning.LightningModule):
        appearance_embedding_optimizer, appearance_embedding_scheduler = [], []
        if self.n_appearance_embedding_dims > 0:
            # fmt: off
            appearance_embedding_optimizer = Adam().instantiate(
                [{
                    "params": self.appearance_embedding.parameters(),
                    "lr": self.config.optimization.appearance_embedding_lr_init,
                    "name": "appearance_embedding",
                }],
                lr=0.0,
                eps=self.config.optimization.eps,
            )
            # fmt: on
            appearance_embedding_scheduler = (
                ExponentialDecayScheduler(
                    lr_final=self.config.optimization.appearance_embedding_lr_final,
                    max_steps=self.config.optimization.max_steps,
                )
                .instantiate()
                .get_scheduler(
                    optimizer=appearance_embedding_optimizer,
                    lr_init=self.config.optimization.appearance_embedding_lr_init,
                )
            )
        return appearance_embedding_optimizer, appearance_embedding_scheduler

    def prepare_primitives(self, pc: GridGaussianModel, viewpoint_camera: Camera):
        # filter by level
        anchor_mask, prog_ratio, transition_mask = [None] * 3

        if getattr(pc, "get_levels", None) is not None and pc.get_levels.shape[0] > 0:
            pc: LoDGridGaussianModel
            anchor_mask, prog_ratio, transition_mask = pc.filter_anchor_by_level(viewpoint_camera)

        # filter by preprojection
        anchor_mask = pc.filter_anchor_by_preprojection(viewpoint_camera, anchor_mask)

        # scaffold model
        # if isinstance(pc, ScaffoldGaussianModelMixin):
        if getattr(pc, "gaussian_mlps", None) is not None and getattr(pc, "get_anchor_features", None) is not None:
            pc: ScaffoldGaussianModelMixin
            if self.n_appearance_embedding_dims > 0:
                appearance_code = self.appearance_embedding(viewpoint_camera.appearance_id)
            else:
                appearance_code = None
            return pc.calculate_implicit_properties(
                viewpoint_camera,
                appearance_code=appearance_code,
                anchor_mask=anchor_mask,
                prog_ratio=prog_ratio,
                transition_mask=transition_mask,
            )
        # TODO elif explicit model
        else:
            raise ValueError("Unsupported gaussian model type")

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: GridGaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        render_types: list = None,
        **kwargs,
    ):
        xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask = self.prepare_primitives(
            pc, viewpoint_camera
        )

        # fmt: off
        # +--------------------------------------------------+
        # | modified from `GSplatV1RendererModule.forward()` |
        # +--------------------------------------------------+
        render_type_bits = self.parse_render_types(render_types)
        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)

        # 1. get scales and then project
        if scaling_modifier != 1.0:
            scales = scales * scaling_modifier
        
        projections = GSplatV1.project(
            preprocessed_camera,
            xyz,
            scales,
            rots,
            eps2d=self.config.filter_2d_kernel_size,
            anti_aliased=self.config.anti_aliased,
            radius_clip=self.runtime_options.radius_clip,
            # radius_clip_from=self.runtime_options.radius_clip_from,
            camera_model=self.runtime_options.camera_model,
        )
        radii, means2d, depths, conics, compensations = projections

        radii_squeezed = radii.squeeze(0)
        visibility_filter = radii_squeezed > 0

        # 2. get opacities and then isect encoding
        opacities = opacities.unsqueeze(0)  # [1, N]
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

        def rasterize(input_features: torch.Tensor, background, return_alpha: bool = False):
            rendered_colors, rendered_alphas = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities,
                colors=input_features,
                background=background,
                tile_size=self.config.block_size,
            )

            if return_alpha:
                return rendered_colors, rendered_alphas.squeeze(0).squeeze(-1)
            return rendered_colors

        # rgb
        rgb = None
        acc_vis = None
        if self.is_type_required(render_type_bits, self._RGB_REQUIRED):
            if pc.config.color_mode == 'SHs':
                viewdirs = xyz.detach() - viewpoint_camera.camera_center
                colors = spherical_harmonics(pc.activate_sh_degree, viewdirs, colors, visibility_filter)
            rgb = rasterize(colors, bg_color).permute(2, 0, 1)
            # avoid overriding by hard depth
            acc_vis = means2d.has_hit_any_pixels

        alpha = None
        acc_depth_im = None
        acc_depth_inverted_im = None
        exp_depth_im = None
        exp_depth_inverted_im = None
        inv_depth_alt = None
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            # acc depth
            acc_depth_im, alpha = rasterize(depths[0].unsqueeze(-1), torch.zeros((1,), device=bg_color.device), True)
            alpha = alpha[..., None]

            # acc depth inverted
            if self.is_type_required(render_type_bits, self._ACC_DEPTH_INVERTED_REQUIRED):
                acc_depth_inverted_im = torch.where(acc_depth_im > 0, 1.0 / acc_depth_im, acc_depth_im.detach().max())
                acc_depth_inverted_im = acc_depth_inverted_im.permute(2, 0, 1)

            # exp depth
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_REQUIRED):
                exp_depth_im = torch.where(alpha > 0, acc_depth_im / alpha, acc_depth_im.detach().max())

                exp_depth_im = exp_depth_im.permute(2, 0, 1)

            # alpha
            if self.is_type_required(render_type_bits, self._ALPHA_REQUIRED):
                alpha = alpha.permute(2, 0, 1)
            else:
                alpha = None

            # permute acc depth
            acc_depth_im = acc_depth_im.permute(2, 0, 1)

            # exp depth inverted
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_INVERTED_REQUIRED):
                exp_depth_inverted_im = torch.where(exp_depth_im > 0, 1.0 / exp_depth_im, exp_depth_im.detach().max())

        # inverse depth
        inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1.0 / (depths[0].clamp_min(0.0) + 1e-8).unsqueeze(-1)
            inverse_depth_im = rasterize(
                inverse_depth, torch.zeros((1,), dtype=torch.float, device=bg_color.device)
            ).permute(2, 0, 1)
            inv_depth_alt = inverse_depth_im

        # hard depth
        hard_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_DEPTH_REQUIRED):
            hard_depth_im, _ = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities + (1 - opacities.detach()),
                colors=depths[0].unsqueeze(-1),
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                tile_size=self.config.block_size,
            )
            hard_depth_im = hard_depth_im.permute(2, 0, 1)

        # hard inverse depth
        hard_inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1.0 / (depths[0].clamp_min(0.0) + 1e-8).unsqueeze(-1)
            hard_inverse_depth_im, _ = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities + (1 - opacities.detach()),
                colors=inverse_depth,
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                tile_size=self.config.block_size,
            )

            hard_inverse_depth_im = hard_inverse_depth_im.permute(2, 0, 1)
            inv_depth_alt = hard_inverse_depth_im

        return {
            "render": rgb,
            "alpha": alpha,
            "acc_depth": acc_depth_im,
            "acc_depth_inverted": acc_depth_inverted_im,
            "exp_depth": exp_depth_im,
            "exp_depth_inverted": exp_depth_inverted_im,
            "inverse_depth": inverse_depth_im,
            "hard_depth": hard_depth_im,
            "hard_inverse_depth": hard_inverse_depth_im,
            "inv_depth_alt": inv_depth_alt,
            "viewspace_points": means2d,
            "viewspace_points_grad_scale": 0.5
            * torch.tensor([preprocessed_camera[-1]]).to(means2d).clamp_(max=self.config.max_viewspace_grad_scale),
            "visibility_filter": visibility_filter,
            "acc_vis": acc_vis,
            "radii": radii_squeezed,
            "xyz": xyz,
            "scales": scales,
            "rotations": rots,
            "opacities": opacities[0],
            "projections": projections,
            "isects": isects,
            "conics": conics,
            # extra infos
            "anchor_mask": anchor_mask,
            "primitive_mask": primitive_mask,
        }
        # fmt: on

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        super().setup_web_viewer_tabs(viewer, server, tabs)

        # if isinstance(viewer.gaussian_model, LoDGridGaussianModel):
        if (
            getattr(viewer.gaussian_model, "get_levels", None) is not None
            and viewer.gaussian_model.get_levels.shape[0] > 0
        ):
            with tabs.add_tab("Octree"):
                self._lod_options = ViewerOptions(viewer, server)


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
            viewer.viewer_renderer.gaussian_model.set_activate_level(self.activate_level_slider.value)
            viewer.rerender_for_all_client()
