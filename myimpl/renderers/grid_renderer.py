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
from myimpl.utils.multiview_loss import MultiViewLossUtils

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

    render_feature_size: int = 256
    """short side of the feature map"""

    appearance_model: AppearanceModelConfig = field(default_factory=lambda: AppearanceModelConfig())

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return GridGaussianRendererModule(self)


class GridGaussianRendererModule(GSplatV1RendererModule):
    _FEATURE_REQUIRED = 1 << 10
    _PSEUDO_VIEW_REQUIRED = 1 << 11
    RENDER_TYPE_BITS = {
        **GSplatV1RendererModule.RENDER_TYPE_BITS,
        "acc_depth": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,
        "inverse_depth": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,
        "normal": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,
        "normal_from_depth": GSplatV1RendererModule._ACC_DEPTH_REQUIRED,  # all images rendered if depth is required
        "feature": _FEATURE_REQUIRED,
        "pseudo_view": _PSEUDO_VIEW_REQUIRED,
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

        if lightning_module is not None:
            if "pseudo_view" in lightning_module.hparams["renderer_output_types"]:
                self.multiview_from_iter = lightning_module.metric.config.multiview_from_iter
                # compute disturb
                from simple_knn._C import distCUDA2

                cam_centers = lightning_module.trainer.datamodule.dataparser_outputs.train_set.cameras.camera_center
                dist2 = torch.clamp_min(distCUDA2(cam_centers.cuda()), 0.0000001)
                self.disturb = torch.median(torch.sqrt(dist2)) * 0.5 * math.sqrt(0.5)

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

    def training_forward(self, step, module, viewpoint_camera, pc, bg_color, render_types=None, **kwargs):
        output_pkg = self(viewpoint_camera, pc, bg_color, render_types=render_types, **kwargs)

        render_type_bits = self.parse_render_types(render_types)
        if self.is_type_required(render_type_bits, self._PSEUDO_VIEW_REQUIRED) and step + 1 >= getattr(
            self, "multiview_from_iter", 1 << 30
        ):
            viewpoint_camera = GridRendererUtils.batch_cameras(viewpoint_camera)
            pseudo_view = MultiViewLossUtils.get_pseudo_view(viewpoint_camera, output_pkg["acc_depth"], self.disturb)
            # pseudo_view = MultiViewLossUtils.get_pseudo_view(viewpoint_camera, output_pkg["acc_depth"], 0.0)
            pseudo_render = self(pseudo_view, pc, bg_color, render_types=render_types, **kwargs)
            output_pkg.update({"pseudo_results": {"view": pseudo_view, "render": pseudo_render}})
        return output_pkg

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
        # iterate camera to calculate primitives' properties and project to plane
        primitives = GridRendererUtils.prepare_primitives_loop(
            viewpoint_camera, pc, getattr(self, "appearance_embedding", None), **kwargs
        )
        projections_list, isects_list, visibility_filter, preprocessed_camera = (
            GridRendererUtils.project_to_pixels_loop(
                self,
                primitives,
                viewpoint_camera,
                scaling_modifier=scaling_modifier,
                return_preprocessed_cam=True,
                **kwargs,
            )
        )
        viewmats, Ks, (image_width, image_height) = preprocessed_camera
        properties_list = [(*p, v) for p, v in zip(primitives, visibility_filter)]

        # concatenate projections from cameras for parallel rasterization
        projections = GridRendererUtils.concatenate_projections(projections_list, isects_list)
        means2d, *_ = projections

        # prepare feature to be rasterized
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
        render, alpha = GridRendererUtils.rasterize_cat_projections(
            preprocessed_camera=preprocessed_camera,
            projections=projections,
            properties=(input_colors, input_opacities),
            bg_color=bg_color,
            tile_size=self.config.block_size,
        )

        # gather render results
        rgb = render[..., :3].permute(0, 3, 1, 2).squeeze(0)  # [C, 3, H, W] or [3, H, W]
        alpha = alpha.permute(0, 3, 1, 2).squeeze(0)  # [C, 1, H, W] or [1, H, W]

        normal, inv_depth, acc_depth, normal_from_depth, normal_local, plane_dist = [None] * 6
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            c2w = viewmats.inverse()
            pgsr_maps = render[..., 3:]  # [C, H, W, 5]
            normal = F.normalize(pgsr_maps[..., :3], dim=-1)
            inv_depth = pgsr_maps[..., 3:4]
            plane_dist = pgsr_maps[..., 4:5]

            normal_local = torch.einsum("...md, ...hwd -> ...hwm", viewmats[..., :3, :3], normal)
            acc_depth = GridRendererUtils.compute_acc_depth(normal_local, plane_dist, Ks=Ks)  # [C, H, W, 1]
            normal_from_depth = depth_to_normal(depths=acc_depth, camtoworlds=c2w, Ks=Ks)

            # change to CDHW mode, and maybe squeeze dim0
            inv_depth = inv_depth.permute(0, 3, 1, 2).squeeze(0)
            acc_depth = acc_depth.permute(0, 3, 1, 2).squeeze(0)
            normal = normal.permute(0, 3, 1, 2).squeeze(0)
            normal_from_depth = normal_from_depth.permute(0, 3, 1, 2).squeeze(0)

        render_feature = None
        if self.is_type_required(render_type_bits, self._FEATURE_REQUIRED):
            render_feature, _ = self.render_feature(
                properties_list,
                viewpoint_camera,
                pc,
                scaling_modifier=scaling_modifier,
                **kwargs,
            )
            render_feature = render_feature.permute(0, 3, 1, 2).squeeze(0)

        output_pkg = {}
        # render output
        output_pkg.update({"render": rgb, "alpha": alpha})
        output_pkg.update(
            {
                # for viewer, [C, D, H, W] or [D, H, W]
                "inverse_depth": inv_depth,
                "acc_depth": acc_depth,
                "normal": normal,
                "normal_from_depth": normal_from_depth,
                # used later, [C, H, W, D]
                "plane_dist": plane_dist,
            }
        )
        output_pkg.update({"render_feature": render_feature})

        # implicit primitives
        output_pkg.update(GridRendererUtils.get_implicit_properties(properties_list))
        # viewspace grad
        grad_scale = (
            torch.tensor([[image_width[0], image_height[0]]])
            .to(means2d)
            .clamp_(max=self.config.max_viewspace_grad_scale)
        )
        output_pkg.update(
            {
                **GridRendererUtils.get_projections(projections),
                "viewspace_points_grad_scale": grad_scale,
            }
        )

        return output_pkg

    def render_feature(
        self,
        properties_list: List[Tuple[torch.Tensor, ...]],
        viewpoint_camera: Cameras,
        pc: GridGaussianModel,
        scaling_modifier=1.0,
        **kwargs,
    ):
        projections_list, isects_list, visibility_filter, preprocessed_camera = (
            GridRendererUtils.project_to_pixels_loop(
                self,
                properties_list,
                viewpoint_camera,
                scaling_modifier=scaling_modifier,
                return_preprocessed_cam=True,
                **kwargs,
            )
        )
        projections = GridRendererUtils.concatenate_projections(projections_list, isects_list)
        means2d, *_ = projections

        input_features = means2d.new_zeros((0, pc.config.feature_dim))
        input_opacities = means2d.new_zeros((0,))
        for cam_id in range(len(viewpoint_camera)):
            _, _, _, _, _opacities, anchor_mask, primitive_mask, _visibility_filter = properties_list[cam_id]
            features = repeat(pc.get_anchor_features[anchor_mask], "n d -> (n o) d", o=pc.n_offsets)
            features = features[primitive_mask]
            features = features[visibility_filter]

            input_opacities = torch.cat([input_opacities, _opacities[visibility_filter]], dim=0)
            input_features = torch.cat([input_features, features], dim=0)

        render_feature, alpha = GridRendererUtils.rasterize_cat_projections(
            preprocessed_camera=preprocessed_camera,
            projections=projections,
            properties=(input_features, input_opacities),
            bg_color=means2d.new_zeros((len(viewpoint_camera), pc.config.feature_dim)),
            tile_size=self.config.block_size,
        )
        return render_feature, alpha

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
            "feature": RendererOutputInfo("render_feature", type=RendererOutputTypes.FEATURE_MAP),
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

        return viewmats, Ks, (width.tolist(), height.tolist())

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
    def prepare_primitives_loop(
        cls,
        viewpoint_camera: Cameras,
        pc: GridGaussianModel,
        appearance_embedding: Optional[nn.Embedding] = None,
        **kwargs,
    ) -> list:
        primitives = []
        for cam in viewpoint_camera:
            xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask = cls.prepare_primitives(
                pc, cam, appearance_embedding, **kwargs
            )
            primitives.append((xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask))

        return primitives  # xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask

    @classmethod
    def project_to_pixels(
        cls,
        renderer: GridGaussianRendererModule,
        xyz: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        preprocessed_camera: tuple,
        scaling_modifier: float = 1.0,
        **kwargs,
    ):
        viewmat, intrinsics, (image_width, image_height) = preprocessed_camera
        if scaling_modifier != 1.0:
            scales = scales * scaling_modifier

        cids, gids, radii, means2d, depths, conics, compensations = fully_fused_projection(
            means=xyz,
            covars=None,
            quats=rotations,
            scales=scales,
            viewmats=viewmat.unsqueeze(0),
            Ks=intrinsics.unsqueeze(0),
            width=image_width,
            height=image_height,
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
        tile_width = math.ceil(float(image_width) / float(renderer.config.block_size))
        tile_height = math.ceil(float(image_height) / float(renderer.config.block_size))
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
    def project_to_pixels_loop(
        cls,
        renderer: GridGaussianRendererModule,
        primitives: list,
        viewpoint_camera: Cameras,
        short_length: Optional[int] = None,
        scaling_modifier: float = 1.0,
        return_preprocessed_cam: bool = False,
        **kwargs,
    ):
        projections_list, isects_list, visibility_filter = [], [], []

        preprocessed_camera = cls.preprocess_cameras(viewpoint_camera, short_length=short_length)
        viewmats, Ks, (image_width, image_height) = preprocessed_camera

        for cid in range(len(primitives)):
            _xyz, _scales, _rots, *_ = primitives[cid]

            projections, isects, vis = cls.project_to_pixels(
                renderer,
                _xyz,
                _scales,
                _rots,
                preprocessed_camera=(viewmats[cid], Ks[cid], (image_width[cid], image_height[cid])),
                scaling_modifier=scaling_modifier,
                **kwargs,
            )

            projections_list.append(projections)
            isects_list.append(isects)
            visibility_filter.append(vis)

        outputs = projections_list, isects_list, visibility_filter
        if return_preprocessed_cam:
            outputs = (*outputs, preprocessed_camera)
        return outputs

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
    def rasterize_cat_projections(
        cls,
        preprocessed_camera: tuple,
        projections: tuple,
        properties: tuple,
        bg_color: torch.Tensor,
        tile_size: int = 16,
    ):
        _, _, (image_width, image_height) = preprocessed_camera
        means2d, conics, isects_offsets, flatten_ids = projections
        colors, opacities = properties
        render, alpha = rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            colors=colors,
            opacities=opacities,
            image_width=image_width[0],
            image_height=image_height[0],
            tile_size=tile_size,
            isect_offsets=isects_offsets,
            flatten_ids=flatten_ids,
            backgrounds=bg_color,
            packed=True,
        )
        return render, alpha

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

        normal_global = cls.get_normals(xyz, scales, rotations, viewpoint_cam, **kwargs)

        normal_local = normal_global @ viewpoint_cam.world_to_camera[:3, :3]
        pts_in_cam = xyz @ viewpoint_cam.world_to_camera[:3, :3] + viewpoint_cam.world_to_camera[-1, :3]
        depth_z = pts_in_cam[:, 2]
        local_dist = (normal_local * pts_in_cam).sum(-1).abs()
        depth_feats = xyz.new_zeros((xyz.shape[0], 5))
        depth_feats[:, :3] = normal_global
        depth_feats[:, 3] = 1.0 / (depth_z.clamp_min(0.0) + 1e-8)
        depth_feats[:, 4] = local_dist

        return depth_feats

    @classmethod
    def compute_acc_depth(cls, normal_local: torch.Tensor, plane_dist: torch.Tensor, Ks: torch.Tensor):
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
    def get_implicit_properties(cls, properties_list: list):
        # output pkg
        xyz, scales, rots, _, opacities, anchor_mask, primitive_mask, visibility_filter = zip(*properties_list)
        xyz = torch.cat(xyz, dim=0)  # [N, 3]
        scales = torch.cat(scales, dim=0)
        rots = torch.cat(rots, dim=0)
        opacities = torch.cat(opacities, dim=0)
        anchor_mask = torch.cat([torch.nonzero(mask, as_tuple=False).squeeze() for mask in anchor_mask], dim=0)
        primitive_mask = torch.cat(primitive_mask, dim=0)
        visibility_filter = torch.cat(visibility_filter, dim=0)

        return {
            "xyz": xyz,
            "scales": scales,
            "rotations": rots,
            "opacities": opacities,
            "visibility_filter": visibility_filter,
            "anchor_mask": anchor_mask,
            "primitive_mask": primitive_mask,
        }

    @classmethod
    def get_projections(cls, projections: tuple):
        means2d, conics, isects_offsets, flatten_ids = projections
        return {
            "viewspace_points": means2d,
            "conics": conics,
            "isects": (isects_offsets, flatten_ids),
        }


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
