import torch
from gsplat.cuda._wrapper import rasterize_to_indices_in_range
from gsplat.hit_pixel_count import hit_pixel_count
from gsplat.rasterize_to_visibilities import rasterize_to_visibilities

from internal.renderers import RendererOutputInfo, RendererOutputTypes
from internal.renderers.gsplat_v1_renderer import GSplatV1
from internal.renderers.gsplat_v1_renderer import \
    GSplatV1Renderer as _GSplatV1Renderer
from internal.renderers.gsplat_v1_renderer import \
    GSplatV1RendererModule as _GSplatV1RendererModule

__all__ = ["GSplatMeans2dTrackRenderer", "GSplatMeans2dTrackRendererModule"]


class GSplatMeans2dTrackRenderer(_GSplatV1Renderer):
    def instantiate(self, *args, **kwargs):
        return GSplatMeans2dTrackRendererModule(self)


class GSplatMeans2dTrackRendererModule(_GSplatV1RendererModule):
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
    _MEANS2D_TRACK_REQUIRED = 1 << 10

    RENDER_TYPE_BITS = {
        "rgb": _RGB_REQUIRED,
        "alpha": _ALPHA_REQUIRED | _ACC_DEPTH_REQUIRED,
        "acc_depth": _ACC_DEPTH_REQUIRED,
        "acc_depth_inverted": _ACC_DEPTH_REQUIRED | _ACC_DEPTH_INVERTED_REQUIRED,
        "exp_depth": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED,
        "exp_depth_inverted": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED | _EXP_DEPTH_INVERTED_REQUIRED,
        "inverse_depth": _INVERSE_DEPTH_REQUIRED,
        "hard_depth": _HARD_DEPTH_REQUIRED,
        "hard_inverse_depth": _HARD_INVERSE_DEPTH_REQUIRED,
        "inv_depth_alt": _DEPTH_ALTERNATIVE,
        "means2d_track": _MEANS2D_TRACK_REQUIRED,
    }

    # fmt: off
    def get_available_outputs(self):
        available_outputs = super().get_available_outputs()

        def no_processing(x, *args, **kwargs):
            return x

        available_outputs.update({
            "means2d_track": RendererOutputInfo(
                "means2d_track", type=RendererOutputTypes.OTHER, visualizer=no_processing)
        })
        return available_outputs

    def forward(self, viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        render_type_bits = self.parse_render_types(render_types)

        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)

        # 1. get scales and then project
        scales, status = self.get_scales(viewpoint_camera, pc, **kwargs)
        if scaling_modifier != 1.:
            scales = scales * scaling_modifier

        projections = GSplatV1.project(
            preprocessed_camera,
            pc.get_means(),
            scales,
            pc.get_rotations(),
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
        opacities, status = self.get_opacities(
            viewpoint_camera,
            pc,
            projections,
            visibility_filter,
            status,
            **kwargs,
        )

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
            rgbs = self.get_rgbs(
                viewpoint_camera,
                pc,
                projections,
                visibility_filter,
                status,
                **kwargs,
            )
            rgb = rasterize(rgbs, bg_color).permute(2, 0, 1)
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
                acc_depth_inverted_im = torch.where(acc_depth_im > 0, 1. / acc_depth_im, acc_depth_im.detach().max())
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
                exp_depth_inverted_im = torch.where(exp_depth_im > 0, 1. / exp_depth_im, exp_depth_im.detach().max())

        # inverse depth
        inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths[0].clamp_min(0.) + 1e-8).unsqueeze(-1)
            inverse_depth_im = rasterize(inverse_depth, torch.zeros((1,), dtype=torch.float, device=bg_color.device)).permute(2, 0, 1)
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
            inverse_depth = 1. / (depths[0].clamp_min(0.) + 1e-8).unsqueeze(-1)
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

        # hit pixel count
        means2d_track = None
        if self.is_type_required(render_type_bits, self._MEANS2D_TRACK_REQUIRED):
            # _, _, flatten_ids, isect_offsets = isects
            # hit_pixel_count_im, opacity_scores, alpha_scores, visibility_scores = rasterize_to_visibilities(
            #     means2d.unsqueeze(0),
            #     conics,
            #     opacities,
            #     preprocessed_camera[2][0],
            #     preprocessed_camera[2][1],
            #     tile_size=self.config.block_size,
            #     flatten_ids=flatten_ids,
            #     isect_offsets=isect_offsets,
            # )
            point_sparsify = 1
            modified_projections = radii, means2d, depths, torch.tensor([[[16., 0., 16.]]]).to(means2d).repeat(1, means2d.shape[0], 1), compensations
            rgb = torch.zeros((means2d.shape[0], 1)).to(means2d)
            bg = torch.zeros((1,)).to(means2d)
            opacity_mask = opacities.new_zeros(opacities.shape)
            opacity_mask[..., ::point_sparsify] = 1.

            with torch.no_grad():
                means2d_track = GSplatV1.rasterize(
                    preprocessed_camera,
                    modified_projections,
                    isects,
                    opacities * opacity_mask,
                    rgb,
                    bg,
                    tile_size=self.config.block_size,
                    absgrad=False
                )[1].permute(2, 0, 1)

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
            "means2d_track": means2d_track,
            "viewspace_points": means2d,
            "viewspace_points_grad_scale": 0.5 * torch.tensor([preprocessed_camera[-1]]).to(means2d).clamp_(max=self.config.max_viewspace_grad_scale),
            "visibility_filter": visibility_filter,
            "acc_vis": acc_vis,
            "radii": radii_squeezed,
            "scales": scales,
            "opacities": opacities[0],
            "projections": projections,
            "isects": isects,
        }
