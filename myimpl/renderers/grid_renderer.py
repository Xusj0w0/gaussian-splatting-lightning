import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import lightning
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from gsplat.cuda._wrapper import (fully_fused_projection, isect_offset_encode,
                                  isect_tiles, rasterize_to_pixels,
                                  spherical_harmonics)
from gsplat.utils import depth_to_normal, depth_to_points
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

from internal.cameras.cameras import Camera, Cameras
from internal.optimizers import Adam
from internal.renderers import RendererOutputInfo, RendererOutputTypes
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule,
                                                   RuntimeOptions)
from internal.schedulers import ExponentialDecayScheduler
from myimpl.models.grid_gaussians import (GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)
from myimpl.models.implicit_grid_gaussian import (ImplicitGridGaussianModel,
                                                  ImplicitLoDGridGaussianModel)
from myimpl.utils.cameras import InstantiatedCameras

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

    render_feature_size: int = 256
    """short side of the feature map"""

    def instantiate(self, *args, **kwargs):
        return GridGaussianRendererModule(self)


class GridGaussianRendererModule(GSplatV1RendererModule):
    _FEATURE_REQUIRED = 1 << 10
    RENDER_TYPE_BITS = {
        **GSplatV1RendererModule.RENDER_TYPE_BITS,
        "acc_depth": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,
        "inverse_depth": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,
        "normal": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,
        "normal_from_depth": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,  # all images rendered if depth is required
        "feature": _FEATURE_REQUIRED,
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

    def forward(
        self,
        viewpoint_camera: Union[Camera, Cameras],
        pc: GridGaussianModel,
        bg_color: torch.Tensor,  # [D, ]
        scaling_modifier=1.0,
        render_types: list = None,
        **kwargs,
    ):
        render_type_bits = self.parse_render_types(render_types)

        viewpoint_camera = GridRendererUtils.batch_cameras(viewpoint_camera)
        viewmats, Ks, image_sizes = GridRendererUtils.preprocess_cameras(viewpoint_camera)

        # iterate camera to calculate primitives' properties and project to plane
        # concatenate projected results for parallel rasterization
        properties_list, projections_list, isects_list = [], [], []
        for cam_id, cam in enumerate(viewpoint_camera):
            xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask = GridRendererUtils.prepare_primitives(
                pc, cam, getattr(self, "appearance_embedding", None), **kwargs
            )
            projections, isects, visibility_filter = GridRendererUtils.project_to_pixels(
                self,
                (xyz, scales, rots),
                viewmat=viewmats[cam_id],
                K=Ks[cam_id],
                image_size=image_sizes[cam_id],
                scaling_modifier=scaling_modifier,
                **kwargs,
            )
            properties_list.append(
                (xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask, visibility_filter)
            )
            projections_list.append(projections)
            isects_list.append(isects)

        means2d, conics, isects_offsets, flatten_ids = GridRendererUtils.concatenate_projections(
            projections_list, isects_list
        )

        # get feature to be rasterized
        color_dim = 8 if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED) else 3
        input_colors = means2d.new_zeros((0, color_dim))
        input_opacities = means2d.new_zeros((0,))
        for cam_id in range(len(viewpoint_camera)):
            cam: Camera = viewpoint_camera[cam_id]
            _xyz, _scales, _rots, _colors, _opacities, _, _, visibility_filter = properties_list[cam_id]

            # maybe convert SHs to RGBs
            if pc.config.color_mode == "SHs":
                viewdirs = _xyz.detach() - cam.camera_center
                _colors = spherical_harmonics(pc.activate_sh_degree, viewdirs, _colors)

            # maybe append depth features
            if color_dim > 3:
                _depth_features = GridRendererUtils.get_depth_features(_xyz, _scales, _rots, cam)
                _colors = torch.cat([_colors, _depth_features], dim=-1)  # [N, 8]

            if visibility_filter.sum() > 0:
                input_colors = torch.cat([input_colors, _colors[visibility_filter]], dim=0)
                input_opacities = torch.cat([input_opacities, _opacities[visibility_filter]], dim=0)

        # rasterize
        if len(bg_color) < color_dim:
            bg_color = torch.cat([bg_color, bg_color.new_zeros((color_dim - len(bg_color),))], dim=0)
        bg_color = bg_color.unsqueeze(0).repeat(len(viewpoint_camera), 1).contiguous()
        render, alpha = rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            colors=input_colors,
            opacities=input_opacities,
            image_width=image_sizes[0][0],
            image_height=image_sizes[0][1],
            tile_size=self.config.block_size,
            isect_offsets=isects_offsets,
            flatten_ids=flatten_ids,
            backgrounds=bg_color,
            packed=True,
        )

        rgb = render[..., :3].permute(0, 3, 1, 2).squeeze(0)  # [C, 3, H, W] or [3, H, W]
        alpha = alpha.permute(0, 3, 1, 2).squeeze(0)  # [C, 1, H, W] or [1, H, W]

        normal, inv_depth, acc_depth, normal_from_depth, pointmap = [None] * 5
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            c2w = viewmats.inverse()
            pgsr_maps = render[..., 3:]  # [C, H, W, 5]
            normal_local = F.normalize(pgsr_maps[..., :3], dim=-1)

            inv_depth = pgsr_maps[..., 3:4]  # [C, H, W, 1]
            acc_depth = GridRendererUtils.calc_acc_depth(normal_local, pgsr_maps[..., 4:5], Ks=Ks)  # [C, H, W, 1]
            normal = torch.einsum("...mk, ...ijk -> ...ijm", c2w[..., :3, :3], normal_local)  # [C, H, W, 3]
            pointmap, normal_from_depth = GridRendererUtils.depth_to_normal(depth=acc_depth, c2w=c2w, Ks=Ks)

            # change to CDHW mode, and maybe squeeze dim0
            inv_depth = inv_depth.permute(0, 3, 1, 2).squeeze(0)
            acc_depth = acc_depth.permute(0, 3, 1, 2).squeeze(0)
            normal = normal.permute(0, 3, 1, 2).squeeze(0)
            normal_from_depth = normal_from_depth.permute(0, 3, 1, 2).squeeze(0)

        # output pkg
        xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask, visibility_filter = zip(*properties_list)
        xyz = torch.cat(xyz, dim=0)  # [N, 3]
        scales = torch.cat(scales, dim=0)
        rots = torch.cat(rots, dim=0)
        colors = torch.cat(colors, dim=0)
        opacities = torch.cat(opacities, dim=0)
        anchor_mask = torch.cat([torch.nonzero(mask, as_tuple=False).squeeze() for mask in anchor_mask], dim=0)
        primitive_mask = torch.cat(primitive_mask, dim=0)
        visibility_filter = torch.cat(visibility_filter, dim=0)

        return {
            "render": rgb,  # [3, H, W]
            "alpha": alpha,  # [1, H, W]
            "normal": normal,
            "normal_from_depth": normal_from_depth,
            "acc_depth": acc_depth,
            "inverse_depth": inv_depth,
            "pointmap": pointmap,
            # intermediates
            "viewspace_points": means2d,
            "conics": conics,
            "viewspace_points_grad_scale": 0.5
            * torch.tensor([image_sizes[0]]).to(means2d).clamp_(max=self.config.max_viewspace_grad_scale),
            "visibility_filter": visibility_filter,
            # "acc_vis": acc_vis,
            # "radii": radii_squeezed,
            "xyz": xyz,
            "scales": scales,
            "rotations": rots,
            "opacities": opacities,
            "projections": projections,
            "isects": isects,
            # extra infos
            "anchor_mask": anchor_mask,
            "primitive_mask": primitive_mask,
        }

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        super().setup_web_viewer_tabs(viewer, server, tabs)

        # if isinstance(viewer.gaussian_model, LoDGridGaussianModel):
        if (
            getattr(viewer.gaussian_model, "get_levels", None) is not None
            and viewer.gaussian_model.get_levels.shape[0] > 0
        ):
            with tabs.add_tab("Octree"):
                self._lod_options = ViewerOptions(viewer, server)

    def get_available_outputs(self):
        return {
            "rgb": RendererOutputInfo("render"),
            "alpha": RendererOutputInfo("alpha", type=RendererOutputTypes.GRAY),
            "acc_depth": RendererOutputInfo("acc_depth", type=RendererOutputTypes.GRAY),
            "inverse_depth": RendererOutputInfo("inverse_depth", type=RendererOutputTypes.GRAY),
            "normal": RendererOutputInfo("normal"),
            "normal_from_depth": RendererOutputInfo("normal_from_depth"),
        }


class GridRendererUtils:
    @classmethod
    def batch_cameras(cls, viewpoint_camera: Camera):
        if len(viewpoint_camera.camera_center.shape) == 1:  # Camera
            params = {}
            for field in InstantiatedCameras.__dataclass_fields__:
                val = getattr(viewpoint_camera, field)
                if isinstance(val, torch.Tensor):
                    val = val.unsqueeze(0)
                params[field] = val
            viewpoint_camera = InstantiatedCameras(**params)
        return viewpoint_camera

    @classmethod
    def preprocess_cameras(cls, viewpoint_cameras: Cameras, short_length: Optional[int] = None):
        viewmats = viewpoint_cameras.world_to_camera.transpose(-1, -2)

        if short_length is not None:
            scale = float(short_length) / torch.minimum(viewpoint_cameras.width, viewpoint_cameras.height)
            width = (viewpoint_cameras.width * scale).int()
            height = (viewpoint_cameras.height * scale).int()
            scale_x = width.float() / viewpoint_cameras.width
            scale_y = height.float() / viewpoint_cameras.height
        else:
            scale_x, scale_y = 1.0, 1.0
            width = viewpoint_cameras.width.int()
            height = viewpoint_cameras.height.int()

        Ks = (
            torch.eye(3, dtype=torch.float, device=viewpoint_cameras.R.device)
            .unsqueeze(0)
            .repeat(len(viewpoint_cameras), 1, 1)
        )
        Ks[..., 0, 0] = viewpoint_cameras.fx * scale_x
        Ks[..., 1, 1] = viewpoint_cameras.fy * scale_y
        Ks[..., 0, 2] = viewpoint_cameras.cx * scale_x
        Ks[..., 1, 2] = viewpoint_cameras.cy * scale_y

        return viewmats, Ks, list((w, h) for w, h in zip(width.tolist(), height.tolist()))

    @classmethod
    def prepare_primitives(
        cls,
        pc: GridGaussianModel,
        viewpoint_camera: Camera,
        appearance_embedding: Optional[nn.Embedding] = None,
        **kwargs,
    ):
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
            if appearance_embedding is not None:
                appearance_code = appearance_embedding(viewpoint_camera.appearance_id)
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

    @classmethod
    def project_to_pixels(
        cls,
        renderer: GridGaussianRendererModule,
        geom_properties: Tuple[torch.Tensor],
        viewmat: torch.Tensor,
        K: torch.Tensor,
        image_size: Tuple[int, int],
        scaling_modifier: float = 1.0,
        **kwargs,
    ):
        xyz, scales, rots = geom_properties
        if scaling_modifier != 1.0:
            scales = scales * scaling_modifier

        cids, gids, radii, means2d, depths, conics, compensations = fully_fused_projection(
            means=xyz,
            covars=None,
            quats=rots,
            scales=scales,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=image_size[0],
            height=image_size[1],
            eps2d=renderer.config.filter_2d_kernel_size,
            radius_clip=renderer.runtime_options.radius_clip,
            camera_model=renderer.runtime_options.camera_model,
            calc_compensations=renderer.config.anti_aliased,
            packed=True,
            **kwargs,
        )

        # only id in gids are valid, create visibility filter
        visibility_filter = xyz.new_zeros((xyz.shape[0],), dtype=torch.bool)
        visibility_filter[gids] = True
        if renderer.config.anti_aliased:
            opacities = opacities * compensations
        tile_width = math.ceil(float(image_size[0]) / float(renderer.config.block_size))
        tile_height = math.ceil(float(image_size[1]) / float(renderer.config.block_size))
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d=means2d,
            radii=radii,
            depths=depths,
            tile_size=renderer.config.block_size,
            tile_width=tile_width,
            tile_height=tile_height,
            packed=True,
            n_cameras=1,
            camera_ids=cids,
            gaussian_ids=gids,
        )
        isect_offsets = isect_offset_encode(
            isect_ids=isect_ids, n_cameras=1, tile_width=tile_width, tile_height=tile_height
        )

        projections = (radii, means2d, depths, conics, compensations)
        isects = (tiles_per_gauss, isect_ids, flatten_ids, isect_offsets)
        return projections, isects, visibility_filter

    @classmethod
    def concatenate_projections(cls, projections_list: list, isects_list: list):
        accum_primitives, accum_isects = 0, 0

        # fetch first element
        _, _means2d, _, _conics, _ = projections_list[0]
        _, _, _flatten_ids, _isect_offsets = isects_list[0]

        # create empty tensors
        means2d, conics = _means2d.new_empty((0, 2)), _conics.new_empty((0, 3))
        flatten_ids, isects_offsets = _flatten_ids.new_empty((0,)), _isect_offsets.new_empty(
            (0, *_isect_offsets.shape[-2:])
        )

        for projections, isects in zip(projections_list, isects_list):
            _, _means2d, _, _conics, _ = projections
            _, _, _flatten_ids, _isect_offsets = isects

            # concatenate
            means2d = torch.cat([means2d, _means2d], dim=0)
            conics = torch.cat([conics, _conics], dim=0)
            # isect idx to gaussian idx
            flatten_ids = torch.cat([flatten_ids, _flatten_ids + accum_primitives], dim=0)
            # tile's start isect idx
            isects_offsets = torch.cat([isects_offsets, _isect_offsets + accum_isects], dim=0)

            accum_primitives += _means2d.shape[0]
            accum_isects += _flatten_ids.shape[0]

        return means2d, conics, isects_offsets, flatten_ids

    @classmethod
    def get_depth_features(
        cls, xyz: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, viewpoint_cam: Camera, **kwargs
    ) -> torch.Tensor:
        """
        Return:
        - **pgsr_props**: [N, 5].
            - 0-3 ~ normals in camera coordinates
            - 4 ~ invdepths
            - 5 ~ distances to the plane defined by gaussians.
        """

        # rotation_matrices = quaternion_to_matrix(rotations)  # [N, 3, 3]
        # smallest_scale_idx = scales.argmin(dim=-1)[..., None, None].expand(-1, 3, -1)
        # normal_global = rotation_matrices.gather(2, smallest_scale_idx).squeeze(dim=-1)  # [N, 3]

        # gaussian_to_cam_global = viewpoint_cam.camera_center - xyz  # [N, 3]
        # neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        # normal_global[neg_mask] = -normal_global[neg_mask]
        normal_global = cls.get_normals(xyz, scales, rotations, viewpoint_cam, **kwargs)

        local_normal = normal_global @ viewpoint_cam.world_to_camera[:3, :3]
        pts_in_cam = xyz @ viewpoint_cam.world_to_camera[:3, :3] + viewpoint_cam.world_to_camera[-1, :3]
        depth_z = pts_in_cam[:, 2]
        local_dist = (local_normal * pts_in_cam).sum(-1).abs()
        depth_feats = xyz.new_zeros((xyz.shape[0], 5))
        depth_feats[:, :3] = local_normal
        depth_feats[:, 3] = 1.0 / (depth_z.clamp_min(0.0) + 1e-8)
        depth_feats[:, 4] = local_dist

        return depth_feats

    @classmethod
    def calc_acc_depth(cls, normal_local: torch.Tensor, plane_dist: torch.Tensor, Ks: torch.Tensor):
        """
        :Args:
        - **normal_local**: [C, H, W, 3]
        - **plane_dist**: [C, H, W, 1]
        - **Ks**: [1, 3, 3], all normal_local and plane_dist share the same camera intrinsics

        :Returns:
        - **depth**: [C, H, W, 1]
        """

        H, W = normal_local.shape[1:3]
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W).to(normal_local), torch.arange(H).to(normal_local), indexing="xy"
        )
        points = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        offsets = Ks[..., :2, -1].unsqueeze(-2).unsqueeze(-2)  # [1, 1, 2]
        scales = 1.0 / torch.diagonal(Ks[..., :2, :2], dim1=-2, dim2=-1).unsqueeze(-2).unsqueeze(-2)  # [21]
        coordinates = (points - offsets) * scales  # [H, W, 2]

        cosine = -(coordinates * normal_local[..., :2] + normal_local[..., -1:]).sum(
            dim=-1, keepdim=True
        )  # [C, H, W, 1]
        depth = plane_dist / (cosine + 1e-8)
        return depth

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

    @classmethod
    def depth_to_normal(cls, depth: torch.Tensor, c2w: torch.Tensor, Ks: torch.Tensor):
        """
        :Args:
        - **depth**: [..., H, W, 1]
        - **c2w**: [..., 4, 4]
        - **Ks**: [..., 3, 3]

        :Returns:
        - **points**: [..., H, W, 3]
        - **normal**: [..., H, W, 3]
        """
        points = depth_to_points(depth, c2w, Ks, True)  # [..., H, W, 3]
        dx = torch.cat([points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=-3)  # [..., H-2, W-2, 3]
        dy = torch.cat([points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=-2)  # [..., H-2, W-2, 3]
        normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)  # [..., H-2, W-2, 3]
        normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)  # [..., H, W, 3]
        return points, normals


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
