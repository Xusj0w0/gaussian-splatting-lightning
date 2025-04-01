from types import MethodType

import torch
from gsplat.hit_pixel_count import hit_pixel_count

import internal.renderers as renderers
from internal.cameras.cameras import Camera
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers.gsplat_v1_renderer import (GSplatV1,
                                                   GSplatV1RendererModule)
from internal.utils.visualizers import Visualizers
from internal.viewer.renderer import ViewerRenderer as _ViewerRenderer
from large_scene.utils.partition_lod_renderer import PartitionLoDRendererModule

__all__ = ["ViewerRenderer"]
# fmt: off
class ViewerRenderer(_ViewerRenderer):
    def __init__(
            self,
            gaussian_model,
            renderer: renderers.Renderer,
            background_color,
    ):
        super().__init__(gaussian_model, renderer, background_color)
        self.max_count = 1.

    def setup_options(self, viewer, server):
        available_outputs = self.renderer.get_available_outputs()
        first_type_name = list(available_outputs.keys())[0]

        with server.gui.add_folder("Output"):
            # setup output type dropdown
            output_type_dropdown = server.gui.add_dropdown(
                label="Type",
                options=list(available_outputs.keys()),
                initial_value=first_type_name,
            )
            self.output_type_dropdown = output_type_dropdown

            @output_type_dropdown.on_update
            def _(event):
                if event.client is None:
                    return
                with server.atomic():
                    # whether valid type
                    output_type_info = available_outputs.get(output_type_dropdown.value, None)
                    if output_type_info is None:
                        return

                    self._set_output_type(output_type_dropdown.value, output_type_info)

                    viewer.rerender_for_all_client()

            self._setup_depth_map_options(viewer, server)

        # update default output type to the first one, must be placed after gui setup
        self._set_output_type(name=first_type_name, renderer_output_info=available_outputs[first_type_name])

    def _set_output_type(self, name, renderer_output_info):
        """
        Update properties
        """
        # toggle depth map option, only enable when type is `gray` and `visualizer` is None
        self._set_depth_map_option_visibility(renderer_output_info.type == renderers.RendererOutputTypes.GRAY and renderer_output_info.visualizer is None)

        # set visualizer
        visualizer = renderer_output_info.visualizer
        if visualizer is None:
            if renderer_output_info.type == renderers.RendererOutputTypes.RGB:
                visualizer = self.no_processing
            elif renderer_output_info.type == renderers.RendererOutputTypes.GRAY:
                visualizer = self.depth_map_processor
            elif renderer_output_info.type == renderers.RendererOutputTypes.NORMAL_MAP:
                visualizer = self.normal_map_processor
            elif renderer_output_info.type == renderers.RendererOutputTypes.FEATURE_MAP:
                visualizer = self.feature_map_processor
            else:
                raise ValueError(f"Unsupported output type `{renderer_output_info.type}`")
        else:
            if renderer_output_info.type == renderers.RendererOutputTypes.OTHER and renderer_output_info.key == "means2d_track":
                # similar to depth map visualization
                self._set_depth_map_option_visibility(True)
                visualizer = self.means2d_track_processor
            else:
                raise ValueError(f"Unsupported output type `{renderer_output_info.type}` with visualizer `{visualizer}`")

        # update
        self.set_output_info(name, renderer_output_info, visualizer)

    # def get_outputs(self, camera, scaling_modifier: float = 1.):
    #     render_type, output_info, output_processor = self.output_info
    #     if render_type == "hit_pixel_count":
    #         render_outputs = self.renderer.render_hit_pixel_count(self.gaussian_model, camera, scaling_modifier)
    #         image = output_processor(render_outputs[output_info.key], render_outputs, output_info)
    #         if image.shape[0] == 1:
    #             image = image.repeat(3, 1, 1)
    #     else:
    #         image = super().get_outputs(camera, scaling_modifier)
    #     return image

    def means2d_track_processor(self, means2d_track_map, *args, **kwargs):
        return Visualizers.float_colormap(means2d_track_map, self.depth_map_color_map)
