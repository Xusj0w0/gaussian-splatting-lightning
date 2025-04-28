from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightning
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from gsplat.cuda._wrapper import rasterize_to_pixels, spherical_harmonics
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

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
    _PGSR_DEPTH_REQUIRED = 1 << 10
    RENDER_TYPE_BITS = {
        **GSplatV1RendererModule.RENDER_TYPE_BITS,
        "pgsr_depth": _PGSR_DEPTH_REQUIRED,
    }

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

    def prepare_primitives(self, pc: GridGaussianModel, viewpoint_camera: Camera, **kwargs):
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
                **kwargs,
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
            pc, viewpoint_camera, **kwargs
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
        # radii: [C, N], means2d: [C, N, 2], depths: [C, N], conics: [C, N, 3], compensations: [C, N]
        radii, means2d, depths, conics, compensations = projections

        radii_squeezed = radii.squeeze(0)
        visibility_filter = radii_squeezed > 0

        # 2. get opacities and then isect encoding
        # extend camera dim
        opacities = opacities.unsqueeze(0)  # opacities: [C, N]
        colors = colors.unsqueeze(0)
        bg_color = bg_color.unsqueeze(0)

        if self.config.anti_aliased:
            opacities = opacities * compensations

        isects = self.isect_encode(
            preprocessed_camera,
            projections,
            opacities,
            tile_size=self.config.block_size,
        )

        # 3. rasterization
        # fmt: on
        if pc.config.color_mode == "SHs":
            viewdirs = xyz.detach() - viewpoint_camera.camera_center
            colors = spherical_harmonics(pc.activate_sh_degree, viewdirs, colors, visibility_filter)

        if self.is_type_required(render_type_bits, self._PGSR_DEPTH_REQUIRED):
            pgsr_props = self.get_pgsr_props(xyz, scales, rots, viewpoint_camera)
            colors = torch.cat([colors, pgsr_props], dim=-1)  # [N, 8]
            bg_color = torch.cat([bg_color, bg_color.new_zeros((pgsr_props.shape[0], pgsr_props.shape[-1]))], dim=-1)

        # render: [C, H, W, D], alpha_map: [C, H, W, 1]
        render, alpha_map = rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            colors=colors,
            opacities=opacities,
            image_width=preprocessed_camera[-1][0],
            image_height=preprocessed_camera[-1][1],
            tile_size=self.config.block_size,
            isect_offsets=isects[-1],
            flatten_ids=isects[-2],
            backgrounds=bg_color,
        )

        rgb = render[..., :3].permute(0, 3, 1, 2).squeeze(0)
        # acc_vis = means2d.has_hit_any_pixels

        normal_map, invdepth_map, plane_dist_map, unbiased_depth_map, normal_map_from_depth, point_map = [None] * 6
        if self.is_type_required(render_type_bits, self._PGSR_DEPTH_REQUIRED):
            pgsr_maps = render[..., 3:]
            local_normal_map = F.normalize(pgsr_maps[..., :3], dim=-1)  # [C, H, W, 3]
            invdepth_map = pgsr_maps[..., 3]  # [C, H, W]
            plane_dist_map = pgsr_maps[..., 4]  # [C, H, W]
            unbiased_depth_map = self.calc_unbiased_depth_map(local_normal_map, plane_dist_map, preprocessed_camera[1])
            normal_map = torch.einsum(
                "c i j k, c k m -> c i j m", local_normal_map, preprocessed_camera[0][..., :3, :3]
            )
            point_map = self.depths_to_points(viewpoint_camera, unbiased_depth_map)
            normal_map_from_depth = self.points_to_normals(point_map)

            normal_map = normal_map.squeeze(0)
            normal_map_from_depth = normal_map_from_depth.squeeze(0)
            invdepth_map = invdepth_map.squeeze(0)
            unbiased_depth_map = unbiased_depth_map.squeeze(0)

        return {
            "render": rgb,  # [H, W, 3]
            "alpha": alpha_map,  # [1, H, W]
            "normals": normal_map,
            "normals_from_depths": normal_map_from_depth,
            "inverse_depths": invdepth_map,
            "plane_dists": plane_dist_map,
            "depths_unbiased": unbiased_depth_map,
            "points": point_map,
            # intermediates
            "viewspace_points": means2d,
            "viewspace_points_grad_scale": 0.5
            * torch.tensor([preprocessed_camera[-1]]).to(means2d).clamp_(max=self.config.max_viewspace_grad_scale),
            "visibility_filter": visibility_filter,
            # "acc_vis": acc_vis,
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

    def get_pgsr_props(
        self, xyz: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, viewpoint_cam: Camera, **kwargs
    ) -> torch.Tensor:
        """
        Return:
        - **pgsr_props**: [C, N, 5].
            - 0-3 ~ normals in camera coordinates
            - 4 ~ invdepths
            - 5 ~ distances to the plane defined by gaussians.
        """

        rotation_matrices = quaternion_to_matrix(rotations)  # [N, 3, 3]
        smallest_scale_idx = scales.argmin(dim=-1)[..., None, None].expand(-1, 3, -1)
        normal_global = rotation_matrices.gather(2, smallest_scale_idx).squeeze(dim=-1)  # [N, 3]

        gaussian_to_cam_global = viewpoint_cam.camera_center - xyz  # [N, 3]
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]

        local_normal = normal_global @ viewpoint_cam.world_to_camera[:3, :3]
        pts_in_cam = xyz @ viewpoint_cam.world_to_camera[:3, :3] + viewpoint_cam.world_to_camera[-1, :3]
        depth_z = pts_in_cam[:, 2]
        local_dist = (local_normal * pts_in_cam).sum(-1).abs()
        pgsr_props = xyz.new_zeros((xyz.shape[0], 5))
        pgsr_props[:, :3] = local_normal
        pgsr_props[:, 3] = 1.0 / (depth_z.clamp_min(0.0) + 1e-8)
        pgsr_props[:, 4] = local_dist

        return pgsr_props.unsqueeze(0)

    def calc_unbiased_depth_map(self, local_normal_map: torch.Tensor, plane_dist: torch.Tensor, Ks: torch.Tensor):
        # local_normal_map: [C, H, W, 3]
        H, W = local_normal_map.shape[-3:-1]
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W).to(local_normal_map), torch.arange(H).to(local_normal_map), indexing="xy"
        )
        points = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        offsets = Ks[..., :2, -1].unsqueeze(1).unsqueeze(1)
        scales = 1.0 / torch.diagonal(Ks[..., :2, :2], dim1=-2, dim2=-1).unsqueeze(1).unsqueeze(1)
        coordinates = (points - offsets) * scales

        cosine = -(coordinates * local_normal_map[..., :2] + local_normal_map[..., -1:]).sum(dim=-1)
        unbiased_depth_map = plane_dist / (cosine + 1e-8)
        return unbiased_depth_map

    @classmethod
    def get_normals(
        cls, xyz: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, viewpoint_cam: Camera, **kwargs
    ) -> torch.Tensor:
        rotation_matrices = quaternion_to_matrix(rotations)  # [N, 3, 3]
        smallest_scale_idx = scales.argmin(dim=-1)[..., None, None].expand(-1, 3, -1)
        normals = rotation_matrices.gather(2, smallest_scale_idx).squeeze(dim=-1)  # [N, 3]

        gaussian_to_cam_global = viewpoint_cam.camera_center - xyz  # [N, 3]
        neg_mask = (normals * gaussian_to_cam_global).sum(-1) < 0.0
        normals[neg_mask] = -normals[neg_mask]
        return normals

    @staticmethod
    def depths_to_points(view, depthmap):
        c2w = (view.world_to_camera.T).inverse()
        W, H = view.width, view.height
        ndc2pix = torch.tensor([[W / 2, 0, 0, W / 2], [0, H / 2, 0, H / 2], [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ view.full_projection
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T

        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device="cuda").float(), torch.arange(H, device="cuda").float(), indexing="xy"
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
        points = points.reshape(*depthmap.shape, 3)
        return points

    @classmethod
    def points_to_normals(cls, points):
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output

    @classmethod
    def depth_to_normal(cls, view, depth):
        """
        view: view camera
        depth: depthmap
        """
        points = cls.depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
        output = cls.points_to_normals(points)
        return output

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
