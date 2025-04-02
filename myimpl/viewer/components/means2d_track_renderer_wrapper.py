import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning
import torch
import torch.nn as nn

from internal.renderers import (Renderer, RendererOutputInfo,
                                RendererOutputTypes)
from internal.renderers.gsplat_v1_renderer import (GSplatV1, GSplatV1Renderer,
                                                   GSplatV1RendererModule)
from internal.utils.visualizers import Visualizers
from myimpl.renderers.grid_renderer import GridGaussianRendererModule
from myimpl.renderers.partition_lod_renderer import PartitionLoDRendererModule


def return_as_is(x, *args, **kwargs):
    return x


class GSplatMeans2dTrackRendererWrapper(Renderer):
    _MEANS2D_TRACK_REQUIRED = 1 << 10
    RENDER_TYPE_BITS = {
        "means2d_track": _MEANS2D_TRACK_REQUIRED,
    }

    def __init__(self, renderer: GSplatV1RendererModule):
        super().__init__()
        assert isinstance(renderer, GSplatV1RendererModule)
        self._renderer = renderer

    @classmethod
    def wrap_renderer(cls, renderer):
        if isinstance(renderer, PartitionLoDRendererModule):
            renderer.gsplat_renderer = cls(renderer.gsplat_renderer)
            return renderer
        return cls(renderer)

    def parse_render_types(self, render_types: list) -> int:
        _render_types = [i for i in render_types if i in self._renderer.RENDER_TYPE_BITS]
        bits = 0
        if len(_render_types) > 0:
            bits = self._renderer.parse_render_types(_render_types)

        for k, v in self.RENDER_TYPE_BITS.items():
            if k in render_types:
                bits |= v
        return bits

    def is_type_required(self, bits: int, type: int) -> bool:
        if type in self.RENDER_TYPE_BITS:
            return bits & type != 0
        else:
            return self._renderer.is_type_required(bits, type)

    def forward(self, viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs): # fmt: skip
        render_type_bits = self.parse_render_types(render_types)
        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)

        _render_types = [i for i in render_types if i in self._renderer.RENDER_TYPE_BITS]
        output_pkg = self._renderer(
            viewpoint_camera,
            pc,
            bg_color,
            scaling_modifier,
            _render_types,
            **kwargs,
        )

        means2d_track = None
        if self.is_type_required(render_type_bits, self._MEANS2D_TRACK_REQUIRED):
            means2d, isects, opacities = (
                output_pkg["viewspace_points"],
                output_pkg["isects"],
                output_pkg["opacities"].unsqueeze(0),
            )
            conics = torch.tensor([[[16.0, 0.0, 16.0]]]).to(means2d).repeat(1, means2d.shape[0], 1)
            projections = None, means2d, None, conics, None
            rgb = torch.zeros((means2d.shape[0], 1)).to(means2d)
            bg = torch.zeros((1,)).to(means2d)
            with torch.no_grad():
                means2d_track = GSplatV1.rasterize(
                    preprocessed_camera,
                    projections,
                    isects,
                    opacities=opacities,
                    colors=rgb,
                    background=bg,
                    tile_size=self._renderer.config.block_size,
                    absgrad=False,
                )[1].permute(2, 0, 1)

            output_pkg.update({"means2d_track": means2d_track})

        return output_pkg

    def before_training_step(
        self,
        step: int,
        module,
    ):
        self._renderer.before_training_step(step, module)

    def after_training_step(
        self,
        step: int,
        module,
    ):
        self._renderer.after_training_step(step, module)

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        self._renderer.setup(stage, *args, **kwargs)

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[
            Union[
                List[torch.optim.Optimizer],
                torch.optim.Optimizer,
            ]
        ],
        Optional[
            Union[
                List[torch.optim.lr_scheduler.LRScheduler],
                torch.optim.lr_scheduler.LRScheduler,
            ]
        ],
    ]:
        self._renderer.training_setup(module)

    def on_load_checkpoint(self, module, checkpoint):
        self._renderer.on_load_checkpoint(module, checkpoint)

    def get_available_outputs(self):
        available_outputs = self._renderer.get_available_outputs()
        available_outputs.update(
            {
                "means2d_track": RendererOutputInfo(
                    "means2d_track", type=RendererOutputTypes.OTHER, visualizer=return_as_is
                )
            }
        )
        return available_outputs

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        self._renderer.setup_web_viewer_tabs(viewer, server, tabs)
