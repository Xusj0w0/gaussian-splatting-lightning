from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from gsplat import fully_fused_projection_2dgs, rasterize_to_pixels_2dgs
from gsplat.utils import depth_to_normal

from internal.cameras.cameras import Camera, Cameras
from internal.models.gaussian import GaussianModel
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule,
                                                   RuntimeOptions)


@dataclass
class GSplat2DGSRenderer(GSplatV1Renderer):
    def instantiate(self, *args, **kwargs) -> "GSplat2DGSRendererModule":
        return GSplat2DGSRenderer(self)


class GSplat2DGSRendererModule(GSplatV1RendererModule):
    _RGB_REQUIRED = 1
    _ALPHA_REQUIRED = 1 << 1
    _ACC_DEPTH_REQUIRED = 1 << 2
    _ACC_DEPTH_INVERTED_REQUIRED = 1 << 3
    _EXP_DEPTH_REQUIRED = 1 << 4
    _EXP_DEPTH_INVERTED_REQUIRED = 1 << 5
    _INVERSE_DEPTH_REQUIRED = 1 << 6
    _HARD_DEPTH_REQUIRED = 1 << 7
    _HARD_INVERSE_DEPTH_REQUIRED = 1 << 8
    _DEPTH_ALTERNATIVE = 1 << 9
    _NORMAL_REQUIRED = 1 << 10
    _DISTORT_REQUIRED = 1 << 11
    _MEDIAN_DEPTH_REQUIRED = 1 << 12
    _SURF_NORMAL_REQUIRED = 1 << 13

    RENDER_TYPE_BITS = {
        "rgb": _RGB_REQUIRED,
        "alpha": _ALPHA_REQUIRED,
        "acc_depth": _ACC_DEPTH_REQUIRED,
        "acc_depth_inverted": _ACC_DEPTH_REQUIRED | _ACC_DEPTH_INVERTED_REQUIRED,
        "exp_depth": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED,
        "exp_depth_inverted": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED | _EXP_DEPTH_INVERTED_REQUIRED,
        "inverse_depth": _INVERSE_DEPTH_REQUIRED,
        "hard_depth": _HARD_DEPTH_REQUIRED,
        "hard_inverse_depth": _HARD_INVERSE_DEPTH_REQUIRED,
        "inv_depth_alt": _DEPTH_ALTERNATIVE,
        "normal": _NORMAL_REQUIRED,
        "distort": _DISTORT_REQUIRED,
        "median_depth": _MEDIAN_DEPTH_REQUIRED,
        "surf_normal": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED | _SURF_NORMAL_REQUIRED,
    }

    _DEFAULT_RENDER_TYPE = (
        _RGB_REQUIRED
        | _ALPHA_REQUIRED
        | _ACC_DEPTH_REQUIRED
        | _EXP_DEPTH_REQUIRED
        | _NORMAL_REQUIRED
        | _DISTORT_REQUIRED
        | _MEDIAN_DEPTH_REQUIRED
        | _SURF_NORMAL_REQUIRED
    )

    def parse_render_types(self, render_types: Optional[List]) -> int:
        if render_types is None:
            return self._DEFAULT_RENDER_TYPE
        return super().parse_render_types(render_types)

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        render_types: list = None,
        **kwargs,
    ):
        # fmt: off
        render_type_bits = self.parse_render_types(render_types)

        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)

        # 1. get scales and then project
        scales, status = self.get_scales(viewpoint_camera, pc, **kwargs)
        if scaling_modifier != 1.:
            scales = scales * scaling_modifier

        projections = GSplatV1Full.project_2dgs(
            preprocessed_camera,
            pc.get_means(),
            scales,
            pc.get_rotations(),
            eps2d=self.config.filter_2d_kernel_size,
            radius_clip=self.runtime_options.radius_clip,
        )
        radii, means2d, depths, ray_transforms, normals = projections

        radii_squeezed = radii.squeeze(0)
        visibility_filter = radii_squeezed > 0

        # 2. get opacities and then isect encoding
        opacities, status = self.get_opacities(
            viewpoint_camera,
            pc,
            projections,
            visibility_filter,
            status,
            **kwargs,
        )

        opacities = opacities.unsqueeze(0)  # [1, N]

        isects = self.isect_encode(
            preprocessed_camera,
            projections,
            opacities,
            tile_size=self.config.block_size,
        )

        # 3. rasterization
        means2d = means2d.squeeze(0)
        projection_for_rasterization = radii, means2d, depths, ray_transforms, normals

        def rasterize(input_features: torch.Tensor, opacities: torch.Tensor, background, return_extra: bool = False):
            (
                rendered_colors,
                rendered_alphas,
                rendered_normals,
                rendered_distort,
                rendered_median,
            ) = GSplatV1Full.rasterize_2dgs(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities,
                colors=input_features,
                background=background,
                tile_size=self.config.block_size,
            )

            if return_extra:
                return (
                    rendered_colors,
                    rendered_alphas,
                    rendered_normals,
                    rendered_distort,
                    rendered_median,
                )
            return rendered_colors

        extra_rendered = False
        rendered_alpha, rendered_normals, rendered_distort, rendered_median_depth = [None] * 4

        # rgb: use bg_color as background
        rgb = None
        acc_vis = None
        if self.is_type_required(render_type_bits, self._RGB_REQUIRED):
            rgbs = self.get_rgbs(
                viewpoint_camera,
                pc,
                projections,
                visibility_filter,
                status,
                **kwargs,
            )
            if not extra_rendered:
                (
                    rgb,
                    rendered_alpha,
                    rendered_normals,
                    rendered_distort,
                    rendered_median_depth,
                ) = rasterize(rgbs, opacities, bg_color, return_extra=True)
            else:
                rgb = rasterize(rgbs, opacities, bg_color)
            rgb = rgb.permute(2, 0, 1)
            acc_vis = means2d.has_hit_any_pixels

        # depths, including acc depth and inverse depth, use zeros as background
        channel_index = 0
        channel_split_indices = [0]
        input_features = means2d.new_empty((means2d.shape[0], 0))

        # collect depth and inverse depth
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            acc_depth = depths.squeeze(0).unsqueeze(-1)
            input_features = torch.cat([input_features, acc_depth], dim=-1)
            channel_index += acc_depth.shape[-1]
            channel_split_indices.append(channel_index)

        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1.0 / (depths.clamp_min(0.0).squeeze(0).unsqueeze(-1) + 1e-8)
            input_features = torch.cat([input_features, inverse_depth], dim=-1)
            channel_index += inverse_depth.shape[-1]
            channel_split_indices.append(channel_index)

        # render depths
        if not extra_rendered:
            (
                rendered,
                rendered_alpha,
                rendered_normals,
                rendered_distort,
                rendered_median_depth,
            ) = rasterize(rgbs, opacities, bg_color.new_zeros(bg_color.shape), return_extra=True)
        else:
            rendered = rasterize(rgbs, opacities, bg_color.new_zeros(bg_color.shape))
        
        split_index = 0
        # parse depths
        acc_depth_im, acc_depth_inverted_im, exp_depth_im, exp_depth_inverted_im = [None] * 4
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            # acc depth
            acc_depth_im = rendered[..., channel_split_indices[split_index] : channel_split_indices[split_index + 1]]
            
            eps = 1e-8
            # inverted acc depth
            if self.is_type_required(render_type_bits, self._ACC_DEPTH_INVERTED_REQUIRED):
                acc_depth_inverted_im = torch.where(acc_depth_im > eps, 1.0 / acc_depth_im, acc_depth_im.detach().max())
                acc_depth_inverted_im = acc_depth_inverted_im.permute(2, 0, 1)
            # expected depth
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_REQUIRED):
                exp_depth_im = torch.where(rendered_alpha > eps, acc_depth_im / rendered_alpha, acc_depth_im.detach().max())
                exp_depth_im = exp_depth_im.permute(2, 0, 1)
            # inverted expected depth
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_INVERTED_REQUIRED):
                exp_depth_inverted_im = torch.where(exp_depth_im > eps, 1.0 / exp_depth_im, exp_depth_im.detach().max())
            acc_depth_im = acc_depth_im.permute(2, 0, 1)
            split_index += 1

        inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            # inversed depth
            inverse_depth = rendered[..., channel_split_indices[split_index] : channel_split_indices[split_index + 1]]
            inverse_depth_im = inverse_depth.permute(2, 0, 1)
            split_index += 1

        # hard depths, use zeros as background and gradient stopped opacities
        channel_index = 0
        channel_split_indices = [0]
        input_features = means2d.new_empty((means2d.shape[0], 0))

        # collect depth and inverse depth
        if self.is_type_required(render_type_bits, self._HARD_DEPTH_REQUIRED):
            acc_depth = depths.squeeze(0).unsqueeze(-1)
            input_features = torch.cat([input_features, acc_depth], dim=-1)
            channel_index += acc_depth.shape[-1]
            channel_split_indices.append(channel_index)

        if self.is_type_required(render_type_bits, self._HARD_INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1.0 / (depths.clamp_min(0.0).squeeze(0).unsqueeze(-1) + 1e-8)
            input_features = torch.cat([input_features, inverse_depth], dim=-1)
            channel_index += inverse_depth.shape[-1]
            channel_split_indices.append(channel_index)

        # render hard depths
        rendered = rasterize(rgbs, opacities + (1 - opacities.detach()), bg_color.new_zeros(bg_color.shape))

        split_index = 0
        # parse depths
        hard_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_DEPTH_REQUIRED):
            # acc depth
            hard_depth_im = rendered[..., channel_split_indices[split_index] : channel_split_indices[split_index + 1]]
            hard_depth_im = hard_depth_im.permute(2, 0, 1)
            split_index += 1

        hard_inverse_depth_im, inv_depth_alt = None, None
        if self.is_type_required(render_type_bits, self._HARD_INVERSE_DEPTH_REQUIRED):
            # inversed depth
            hard_inverse_depth = rendered[..., channel_split_indices[split_index] : channel_split_indices[split_index + 1]]
            hard_inverse_depth_im = hard_inverse_depth.permute(2, 0, 1)
            inv_depth_alt = hard_inverse_depth_im
            split_index += 1

        # process extra rendered results
        alpha, normal, distort, median_depth, surf_norm = [None] * 5
        if self.is_type_required(render_type_bits, self._ALPHA_REQUIRED):
            alpha = rendered_alpha.permute(2, 0, 1)  # [1, h, w]
        if self.is_type_required(render_type_bits, self._NORMAL_REQUIRED):
            normal = torch.einsum(
                "...ij, ...hwj -> ...hwi",
                torch.linalg.inv(preprocessed_camera[0])[..., :3, :3],
                rendered_normals,
            ).permute(2, 0, 1)  # [3, h, w]
        if self.is_type_required(render_type_bits, self._DISTORT_REQUIRED):
            distort = rendered_distort.permute(2, 0, 1)  # [1, h, w]
        if self.is_type_required(render_type_bits, self._MEDIAN_DEPTH_REQUIRED):
            median_depth = rendered_median_depth.permute(2, 0, 1)  # [1, h, w]
        if self.is_type_required(render_type_bits, self._SURF_NORMAL_REQUIRED):
            surf_norm = depth_to_normal(
                exp_depth_im.permute(1, 2, 0).unsqueeze(0),
                torch.linalg.inv(preprocessed_camera[0]),
                preprocessed_camera[1],
            ).squeeze(0).permute(2, 0, 1)  # [3, h, w]
        # fmt: on

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
            # 2dgs, keep align with internal.renderers.vanilla_2dgs_renderer
            "rend_alpha": alpha,
            "rend_normal": normal,
            "rend_dist": distort,
            "median_depth": median_depth,
            "surf_depth": exp_depth_im,
            "surf_normal": surf_norm,
            # common
            "viewspace_points": means2d,
            "viewspace_points_grad_scale": 0.5
            * torch.tensor([preprocessed_camera[-1]]).to(means2d).clamp_(max=self.config.max_viewspace_grad_scale),
            "visibility_filter": visibility_filter,
            "acc_vis": acc_vis,
            "radii": radii_squeezed,
            "scales": scales,
            "opacities": opacities[0],
            "projections": projections,
            "isects": isects,
        }


class GSplatV1Full(GSplatV1):
    @classmethod
    def project_2dgs(
        cls,
        preprocessed_camera: Tuple,
        means3d: torch.Tensor,  # [N, 3]
        scales: torch.Tensor,  # [N, 3]
        quats: torch.Tensor,  # [N, 4]
        eps2d: float = 0.3,
        **kwargs,
    ):
        """
        Returns:
            A tuple:

            - **radii**. [1, N]
            - **means2d**. [1, N, 2]
            - **depths**. [1, N]
            - **ray_transforms**. [1, N, 3, 3]
            - **normals**. [1, N, 3]
        """
        return fully_fused_projection_2dgs(
            means3d,
            quats,
            scales,
            viewmats=preprocessed_camera[0],
            Ks=preprocessed_camera[1],
            width=preprocessed_camera[2][0],
            height=preprocessed_camera[2][1],
            eps2d=eps2d,
            packed=False,
            **kwargs,
        )

    @classmethod
    def rasterize_2dgs(
        cls,
        preprocessed_camera: Tuple,
        projections,  # NOTE: the means2D must be [N, 2]
        isects,
        opacities: torch.Tensor,  # [1, N]
        colors: torch.Tensor,  # [N, n_color_dims]
        background: torch.Tensor,  # [n_color_dims]
        tile_size: int = 16,
        absgrad: bool = True,
        **kwargs,
    ):
        """
        Returns:
        A tuple:

        - **Rendered colors**.      [image_height, image_width, channels]
        - **Rendered alphas**.      [image_height, image_width, 1]
        - **Rendered normals**.     [image_height, image_width, 3]
        - **Rendered distortion**.  [image_height, image_width, 1]
        - **Rendered median depth**.[image_height, image_width, 1]
        """

        img_width, img_height = preprocessed_camera[-1]
        _, means2d, _, ray_transforms, normals = projections
        _, _, flatten_ids, isect_offsets = isects

        colors = colors.unsqueeze(0)
        background = background.unsqueeze(0)
        densify = means2d.new_zeros(means2d.shape, requires_grad=True)

        (
            rendered_colors,
            rendered_alphas,
            rendered_normals,
            rendered_distort,
            rendered_median,
        ) = rasterize_to_pixels_2dgs(
            means2d=means2d,
            ray_transforms=ray_transforms,
            colors=colors,
            opacities=opacities,
            normals=normals,
            densify=densify,
            image_width=img_width,
            image_height=img_height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            backgrounds=background,
            absgrad=absgrad,
            **kwargs,
        )

        return (
            rendered_colors.squeeze(0),
            rendered_alphas.squeeze(0),
            rendered_normals.squeeze(0),
            rendered_distort.squeeze(0),
            rendered_median.squeeze(0),
        )
