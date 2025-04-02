import io
import traceback
from types import MethodType
from typing import Dict, List, Literal, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import viser

from internal.renderers.renderer import Renderer
from internal.viewer.client import ClientThread
from internal.viewer.viewer import Viewer

__all__ = ["CamPoseController"]


def register_camera_pose_controller():
    def setup_web_viewer_tabs(self, viewer: Viewer, server: viser.ViserServer, tabs: viser.GuiTabGroupHandle):
        setattr(self, "_camera_pose_controller", CamPoseController(viewer, server, tabs))

    Renderer.setup_web_viewer_tabs = MethodType(setup_web_viewer_tabs, Renderer)


class CamPoseController:
    SEPARATOR = " "

    def __init__(self, viewer: Viewer, server: viser.ViserServer, tabs: viser.GuiTabGroupHandle):
        self.viewer = viewer
        self.server = server

        for label, tab_id in zip(tabs._tab_labels, tabs._tab_container_ids):
            if label == "General":
                break

        with server.gui._container_handle_from_uuid[tab_id]:
            with server.gui.add_folder("Camera Pose"):
                self.qvec_text = server.gui.add_text(label="qvec", initial_value="")
                self.pos_text = server.gui.add_text(label="pos", initial_value="")
                self.apply_pose_button = server.gui.add_button(label="Apply pose")
                self.download_image_button = server.gui.add_button(label="Download Image")

        server.on_client_connect(self._register_camera_hook)
        self.apply_pose_button.on_click(self._apply_camera_pose)
        self.download_image_button.on_click(self._send_image_download_request)

    def _register_camera_hook(self, client: viser.ClientHandle):
        client.camera.on_update(self._update_camera_pose_gui)

    def _update_camera_pose_gui(self, event: viser.GuiEvent):
        camera = event.client.camera
        self.qvec_text.value = self.vec2str(camera.wxyz)
        self.pos_text.value = self.vec2str(camera.position)

    def _apply_camera_pose(self, event: viser.GuiEvent):
        with event.client.atomic():
            event.client.camera.wxyz = self.str2vec(self.qvec_text.value)
            event.client.camera.position = self.str2vec(self.pos_text.value)

    def _send_image_download_request(self, event: viser.GuiEvent):
        base64_data = self.render_and_encode(event)
        if base64_data is None:
            return
        filename = "renderer{}".format(".png" if self.viewer.image_format == "png" else ".jpg")
        self.server.send_file_download(filename, base64_data)

    def render_and_encode(self, event: viser.GuiEvent):
        # render image
        output_pkg = self.render_image(event)
        if output_pkg is None:
            return None

        image, jpeg_quality = output_pkg
        _, base64_data = self.encode_image(image, format=self.viewer.image_format, jpeg_quality=jpeg_quality)

        return base64_data

    def render_image(self, event: viser.GuiEvent):
        """modified from internal.viewer.client.ClientThread.render_and_send()"""
        client_thread: ClientThread = self.viewer.clients[event.client.client_id]
        with event.client.atomic():
            max_res, jpeg_quality = client_thread.get_render_options()
            camera = client_thread.get_camera(
                client_thread.client.camera,
                image_size=max_res,
                appearance_id=self.viewer.get_appearance_id_value(),
                time_value=self.viewer.time_slider.value,
                camera_transform=self.viewer.camera_transform,
            ).to_device(self.viewer.device)

        with torch.no_grad():
            try:
                image = client_thread.renderer.get_outputs(camera, scaling_modifier=self.viewer.scaling_modifier.value)
                image = torch.clamp(image, max=1.0)
                image = torch.permute(image, (1, 2, 0))

            except:
                traceback.print_exc()
                return
        return image.cpu().numpy(), jpeg_quality

    def encode_image(self, image: np.ndarray, format: Literal["jpeg", "png"] = "jpeg", jpeg_quality: int = None):
        """modified from viser._scene_api._encode_image_base64()"""
        # to uint8
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            if np.issubdtype(image.dtype, np.integer):
                image = np.clip(image, 0, 255).astype(np.uint8)

        media_type: Literal["image/jpeg", "image/png"]
        with io.BytesIO() as buffer:
            if format == "png":
                media_type = "image/png"
                iio.imwrite(buffer, image, extension=".png")
            elif format == "jpeg":
                media_type = "image/jpeg"
                iio.imwrite(
                    buffer, image[..., :3], extension=".jpg", quality=75 if jpeg_quality is None else jpeg_quality
                )
            else:
                raise ValueError(f"Unsupported format: {format}")

            # base64_data = base64.b64encode(buffer.getvalue()).decode("ascii")
            base64_data = buffer.getvalue()

        return media_type, base64_data

    def vec2str(self, vec: np.ndarray):
        return self.SEPARATOR.join([f"{x:.3f}" for x in vec])

    def str2vec(self, s: str):
        return np.array([float(x) for x in s.split(self.SEPARATOR)])
