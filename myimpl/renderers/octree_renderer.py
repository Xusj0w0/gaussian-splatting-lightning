import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightning
import numpy as np
import torch
import torch.nn.functional as F
from viser import ViserServer, GuiTabGroupHandle

from internal.cameras.cameras import Camera
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule)
from internal.viewer.viewer import Viewer
from myimpl.models.octree_gaussian import OctreeGaussianModel
from myimpl.models.scaffold_gaussian import ScaffoldLoDGaussianModel


@dataclass
class OctreeRenderer(GSplatV1Renderer):
    anti_aliased: bool = False

    def instantiate(self, *args, **kwargs):
        return OctreeRendererModule(self)


class OctreeRendererModule(GSplatV1RendererModule):
    """
    Define anchor filtering functions.
    """

    config: OctreeRenderer

    def get_anchor_mask(self, pc: OctreeGaussianModel, viewpoint_camera: Camera):
        """
        Returns:
            A tuple:
            If `pc.config.dist2level` is "progressive":
            - **anchor_mask**. [n_anchors, ]. Indicating whether the anchor is visible based on viewpoint camera and anchor levels.
            - **prog_ratio**. [n_anchors, ]. Fractional part of predicted level, used for anti-alising cross levels.
            - **transition_mask**. [n_anchors, ]. Indicating whether the level of anchor equals to int level.
            Else:
            - **anchor_mask**. [n_anchors, ]. Indicating whether the anchor is visible based on viewpoint camera and anchor levels.
            - **prog_ratio**. None
            - **transition_mask**. None
        """
        dists = torch.sqrt(torch.sum((pc.get_anchors - viewpoint_camera.camera_center) ** 2, dim=1))
        pred_level = pc.predict_level(dists) + pc.get_extra_levels
        int_level, prog_ratio = pc.map_to_int_level(pred_level, pc.activate_level)
        anchor_mask = pc.get_levels <= int_level

        transition_mask = None
        if pc.config.dist2level == "progressive":
            transition_mask = pc.get_levels == int_level
        return anchor_mask, prog_ratio, transition_mask

    @torch.no_grad()
    def voxel_prefilter(self, pc: OctreeGaussianModel, viewpoint_camera: Camera, anchor_mask: torch.Tensor, **kwargs):
        means = pc.get_anchors[anchor_mask]
        scales = pc.get_scalings[anchor_mask][:, :3]
        quats = means.new_zeros((means.shape[0], 4))
        quats[:, 0] = 1.0

        processed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
        radii = GSplatV1.project(
            processed_camera,
            means3d=means,
            scales=scales,
            quats=quats,
            anti_aliased=False,
        )[0]

        _anchor_mask = anchor_mask.clone()
        _anchor_mask[anchor_mask] = radii.squeeze(0) > 0
        return _anchor_mask

    def calculate_properties(self, viewpoint_camera: Camera, pc: OctreeGaussianModel, *args, **kwargs):
        pass

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: ScaffoldLoDGaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        render_types: list = None,
        **kwargs,
    ):
        pass

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        super().setup_web_viewer_tabs(viewer, server, tabs)

        with tabs.add_tab("LoD"):
            self._lod_options = OctreeLoDOptions(viewer, server)


class OctreeLoDOptions:
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
