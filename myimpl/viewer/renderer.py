import internal.renderers as renderers
from internal.utils.visualizers import Visualizers
from internal.viewer.renderer import ViewerRenderer as _ViewerRenderer
from myimpl.viewer.components.means2d_track_renderer_wrapper import \
    GSplatMeans2dTrackRendererWrapper

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
        self.depth_map_color_map = "gray"
        self.renderer = GSplatMeans2dTrackRendererWrapper.wrap_renderer(self.renderer)

    def _set_output_type(self, name, renderer_output_info):
        # pre process
        if renderer_output_info.key == "means2d_track":
            renderer_output_info.visualizer = self.means2d_track_processor

        super()._set_output_type(name, renderer_output_info)
        # post process
        if renderer_output_info.key == "means2d_track":
            self._set_depth_map_option_visibility(True)

    def means2d_track_processor(self, means2d_track_map, *args, **kwargs):
        return Visualizers.float_colormap(means2d_track_map, self.depth_map_color_map)
