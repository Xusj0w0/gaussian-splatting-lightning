from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightning
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from gsplat import spherical_harmonics

from internal.cameras.cameras import Camera
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule)
from internal.schedulers import ExponentialDecayScheduler
from myimpl.models.grid_gaussians import (GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)
from myimpl.models.implicit_grid_gaussian import (ImplicitGridGaussianModel,
                                                  ImplicitLoDGridGaussianModel)
from myimpl.models.partitionable_implicit_grid_gaussian import (
    PartitionableImplicitGridGaussianModel,
    PartitionableImplicitLoDGridGaussianModel)
from myimpl.renderers.grid_renderer import (GridGaussianRenderer,
                                            GridGaussianRendererModule)


class PartitionGridGaussianRenderer(GridGaussianRenderer):
    pass


class PartitionGridGaussianRendererModule(GridGaussianRendererModule):
    def calculate_implicit_properties(
        self,
        pc: PartitionableImplicitGridGaussianModel,
        viewpoint_camera: Camera,
        anchor_mask: Optional[torch.Tensor] = None,
        prog_ratio: Optional[torch.Tensor] = None,
        transition_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if anchor_mask is None:
            anchor_mask = pc.get_anchors.new_ones((pc.n_anchors,), dtype=torch.bool)
        anchors = pc.get_anchors[anchor_mask]
        features = pc.get_anchor_features[anchor_mask]
        offsets = pc.get_offsets[anchor_mask]
        scalings = pc.get_scalings[anchor_mask]
        anchor_partition_ids = pc.get_anchor_partition_ids[anchor_mask]

        n_anchors, n_offsets = pc.n_anchors, pc.n_offsets

        viewdirs = anchors - viewpoint_camera.camera_center
        viewdirs_norm = torch.norm(viewdirs, dim=1, keepdim=True)
        viewdirs = viewdirs / viewdirs_norm

        if pc.config.use_feature_bank:
            bank_weight = F.softmax(
                pc.forward_by_partition_id(pc.get_feature_bank_mlp, anchor_partition_ids, viewdirs), dim=-1
            ).unsqueeze(dim=1)
            features = features.unsqueeze(dim=-1)
            features = (
                features[:, ::4, :1].repeat(1, 4, 1) * bank_weight[:, :, 0:1]
                + features[:, ::2, :1].repeat(1, 2, 1) * bank_weight[:, :, 1:2]
                + features[:, ::1, :1] * bank_weight[:, :, 2:3]
            )
            features = features.squeeze(dim=-1)
        cat_local_view = torch.cat([features, viewdirs], dim=1)

        opacities = pc.forward_by_partition_id(
            pc.get_opacity_mlp, anchor_partition_ids, features
        )  # try: remove viewdirs
        if prog_ratio is not None and transition_mask is not None:
            prog = prog_ratio[anchor_mask]
            transition = transition_mask[anchor_mask]
            prog[~transition] = 1.0
            opacities = opacities * prog
        opacities = opacities.reshape(-1, 1)

        primitive_mask = (opacities > 0.0).view(-1)

        if self.n_appearance_embedding_dims > 0:
            appearance_code = self.appearance_embedding(viewpoint_camera.appearance_id).view(1, -1).repeat(n_anchors, 1)
            color_input = torch.cat([cat_local_view, appearance_code], dim=-1)
        else:
            color_input = cat_local_view
        colors = pc.forward_by_partition_id(pc.get_color_mlp, anchor_partition_ids, color_input).reshape(-1, 3)

        scale_rots = pc.forward_by_partition_id(pc.get_cov_mlp, anchor_partition_ids, cat_local_view).reshape(-1, 7)

        concatenated = repeat(torch.cat([anchors, scalings], dim=-1), "n c -> (n k) c", k=n_offsets)
        concatenated = torch.cat([concatenated, offsets.reshape(-1, 3), opacities, colors, scale_rots], dim=-1)
        concatenated_masked = concatenated[primitive_mask]
        (
            _anchors,
            _scalings_offset,
            _scalings_scales,
            _offsets,
            _opacities,
            _colors,
            _scales,
            _rots,
        ) = torch.split(concatenated_masked, [3, 3, 3, 3, 1, 3, pc.color_dim, 4], dim=-1)

        xyz = _anchors + _offsets * _scalings_offset
        scales = F.sigmoid(_scales) * _scalings_scales
        rots = pc.rotation_activation(_rots)
        if pc.config.color_mode == "RGB":
            colors = _colors
        elif pc.config.color_mode == "SHs":
            shs = _colors.reshape(-1, int(pc.color_dim // 3), 3)
            viewdirs = xyz.detach() - viewpoint_camera.camera_center
            colors = spherical_harmonics(
                pc.activate_sh_degree,
                viewdirs,
                shs,
            )
        opacities = _opacities.squeeze()

        return xyz, scales, rots, colors, opacities, anchor_mask, primitive_mask
