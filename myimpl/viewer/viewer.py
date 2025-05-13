import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
import viser
import viser.transforms as vtf
import yaml

from internal.renderers import VanillaRenderer
from internal.utils.gaussian_model_editor import MultipleGaussianModelEditor
# from internal.viewer.renderer import ViewerRenderer
from internal.viewer.viewer import Viewer as _Viewer
from myimpl.viewer.components.campose_controller import CamPoseController
from myimpl.viewer.renderer import ViewerRenderer

__all__ = ["Viewer"]
# fmt: off
class Viewer(_Viewer):
    def __init__(
            self,
            model_paths: list[str],
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            sh_degree: int = 3,
            enable_transform: bool = False,
            enable_measurement: bool = False,
            show_cameras: bool = False,
            cameras_json: str = None,
            vanilla_deformable: bool = False,
            vanilla_gs4d: bool = False,
            vanilla_gs2d: bool = False,
            up: list[float] = None,
            default_camera_position: List[float] = None,
            default_camera_look_at: List[float] = None,
            no_edit_panel: bool = False,
            no_render_panel: bool = False,
            gsplat: bool = False,
            gsplat_aa: bool = False,
            gsplat_v1_example: bool = False,
            gsplat_v1_example_aa: bool = False,
            seganygs: str = None,
            vanilla_seganygs: bool = False,
            vanilla_mip: bool = False,
            vanilla_pvg: bool = False,
    ):
        self.device = torch.device("cuda")

        self.model_paths = model_paths
        self.host = host
        self.port = port
        self.background_color = background_color
        self.image_format = image_format
        self.sh_degree = sh_degree
        self.enable_transform = enable_transform
        self.enable_measurement = enable_measurement
        self.show_cameras = show_cameras
        self.extra_video_render_args = []

        self.up_direction = np.asarray([0., 0., 1.])
        self.camera_center = np.asarray([0., 0., 0.])
        self.default_camera_position = default_camera_position
        self.default_camera_look_at = default_camera_look_at

        self.use_gsplat = gsplat
        self.use_gsplat_aa = gsplat_aa

        self.simplified_model = True
        self.show_edit_panel = True
        if no_edit_panel is True:
            self.show_edit_panel = False
        self.show_render_panel = True
        if no_render_panel is True:
            self.show_render_panel = False

        def turn_off_edit_and_video_render_panel():
            self.show_edit_panel = False
            self.show_render_panel = False

        if gsplat_v1_example is True:
            from internal.utils.gaussian_model_loader import \
                GSplatV1ExampleCheckpointLoader
            model, renderer = GSplatV1ExampleCheckpointLoader.load(model_paths[0], self.device, anti_aliased=gsplat_v1_example_aa)
            training_output_base_dir = model_paths[0]
            dataset_type = "Colmap"
        elif vanilla_pvg is True:
            from internal.utils.gaussian_model_loader import \
                VanillaPVGModelLoader
            model, renderer = VanillaPVGModelLoader.search_and_load(model_paths[0], self.device)
            training_output_base_dir = model_paths[0]
            self.checkpoint = None
            dataset_type = "kitti"
            turn_off_edit_and_video_render_panel()
        elif model_paths[0].endswith(".yaml"):
            self.show_edit_panel = False
            self.enable_transform = False
            from internal.models.vanilla_gaussian import VanillaGaussian
            model = VanillaGaussian().instantiate()
            model.setup_from_number(0)
            model.pre_activate_all_properties()
            model.eval()
            # from internal.renderers.partition_lod_renderer import PartitionLoDRenderer
            from myimpl.renderers.partition_lod_renderer import \
                PartitionLoDRenderer
            with open(model_paths[0], "r") as f:
                lod_config = yaml.safe_load(f)
            renderer = PartitionLoDRenderer(**lod_config).instantiate()
            renderer.setup("validation")
            training_output_base_dir = os.getcwd()
            from glob import glob
            appearance_group_ids_file_list = list(glob(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "outputs",
                lod_config["names"][0],
                "**/appearance_group_ids.json",
            ), recursive=True))
            if len(appearance_group_ids_file_list) > 0:
                training_output_base_dir = os.path.dirname(appearance_group_ids_file_list[0])
            dataset_type = "Colmap"
        else:
            load_from = self._search_load_file(model_paths[0])

            # detect whether a SegAnyGS output
            if seganygs is None and load_from.endswith(".ckpt"):
                seganygs_tag_file_path = os.path.join(os.path.dirname(os.path.dirname(load_from)), "seganygs")
                if os.path.exists(seganygs_tag_file_path) is True:
                    print("SegAny Splatting model detected")
                    seganygs = load_from
                    with open(seganygs_tag_file_path, "r") as f:
                        load_from = self._search_load_file(f.read())

            if vanilla_seganygs is True:
                load_from = load_from[:-len(os.path.basename(load_from))]
                load_from = os.path.join(load_from, "scene_point_cloud.ply")

            # whether model is trained by other implementations
            if vanilla_gs4d is True:
                self.simplified_model = False

            # TODO: load multiple models more elegantly
            # load and create models
            model, renderer, training_output_base_dir, dataset_type, self.checkpoint = self._load_model_from_file(load_from)
            if renderer.__class__.__name__ == "GSplatDistributedRendererImpl":
                print(f"[WARNING] You are loading a subset of Gaussians generated by distributed training. If this is not expected, merge them with `utils/merge_distributed_ckpts.py` first.")
                from internal.renderers.gsplat_renderer import GSPlatRenderer
                renderer = GSPlatRenderer()
            # whether a 2DGS model
            if load_from.endswith(".ply") and model.get_scaling.shape[-1] == 2:
                print("2DGS ply detected")
                vanilla_gs2d = True

            def get_load_iteration() -> int:
                return int(os.path.basename(os.path.dirname(load_from)).replace("iteration_", ""))

            # whether model is trained by other implementations
            if vanilla_deformable is True:
                from internal.renderers.vanilla_deformable_renderer import \
                    VanillaDeformableRenderer
                renderer = VanillaDeformableRenderer(
                    os.path.dirname(os.path.dirname(os.path.dirname(load_from))),
                    get_load_iteration(),
                    device=self.device,
                )
                turn_off_edit_and_video_render_panel()
            elif vanilla_gs4d is True:
                from internal.renderers.vanilla_gs4d_renderer import \
                    VanillaGS4DRenderer
                renderer = VanillaGS4DRenderer(
                    os.path.dirname(os.path.dirname(os.path.dirname(load_from))),
                    get_load_iteration(),
                    device=self.device,
                )
                turn_off_edit_and_video_render_panel()
            elif vanilla_gs2d is True:
                from internal.renderers.vanilla_2dgs_renderer import \
                    Vanilla2DGSRenderer
                renderer = Vanilla2DGSRenderer()
                self.extra_video_render_args.append("--vanilla_gs2d")
            elif vanilla_seganygs is True:
                renderer = self._load_vanilla_seganygs(load_from)
                turn_off_edit_and_video_render_panel()
            elif vanilla_mip is True:
                renderer = self._load_vanilla_mip(load_from)
                turn_off_edit_and_video_render_panel()

        # reorient the scene
        cameras_json_path = cameras_json
        if cameras_json_path is None:
            cameras_json_path = os.path.join(training_output_base_dir, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient, dataset_type=dataset_type)
        if up is not None:
            self.camera_transform = torch.eye(4, dtype=torch.float)
            up = torch.tensor(up)
            up = up / torch.linalg.norm(up)
            self.up_direction = up.numpy()

        # load camera poses
        self.camera_poses = self.load_camera_poses(cameras_json_path)
        # calculate camera center
        if len(self.camera_poses) > 0:
            self.camera_center = np.mean(np.asarray([i["position"] for i in self.camera_poses]), axis=0)

        self.available_appearance_options = None

        self.loaded_model_count = 1
        addition_models = model_paths[1:]
        if len(addition_models) == 0:
            # load appearance groups
            appearance_group_filename = os.path.join(training_output_base_dir, "appearance_group_ids.json")
            if os.path.exists(appearance_group_filename) is True:
                with open(appearance_group_filename, "r") as f:
                    self.available_appearance_options = json.load(f)
            # self.available_appearance_options["@Disabled"] = None

            model.freeze()

            if self.show_edit_panel is True or enable_transform is True:
                model = MultipleGaussianModelEditor([model], device=self.device)
        else:
            # switch to vanilla renderer
            model_list = [model.to(torch.device("cpu"))]
            renderer = VanillaRenderer()
            for model_path in addition_models:
                load_from = self._search_load_file(model_path)
                if load_from.endswith(".ckpt"):
                    load_results = self._do_initialize_models_from_checkpoint(load_from, device=torch.device("cpu"))
                else:
                    load_results = self._do_initialize_models_from_point_cloud(load_from, self.sh_degree, device=torch.device("cpu"))
                model_list.append(load_results[0])

            self.loaded_model_count += len(addition_models)

            for i in model_list:
                i.freeze()

            model = MultipleGaussianModelEditor(model_list, device=self.device)

        self.gaussian_model = model

        if seganygs is not None:
            print("loading SegAnyGaussian...")
            renderer = self._load_seganygs(seganygs)
            turn_off_edit_and_video_render_panel()

        # create renderer
        self.viewer_renderer = ViewerRenderer(
            model,
            renderer,
            torch.tensor(background_color, dtype=torch.float, device=self.device),
        )

        self.clients = {}

# fmt: on
    def start(self, block: bool = True, *args, **kwargs):
        super().start(block=False, *args, **kwargs)

        render_folder_id = self.max_res_when_static._impl.parent_container_id
        render_folder_handle: viser.GuiFolderHandle = self._server.gui._container_handle_from_uuid[render_folder_id]
        general_tab_id = render_folder_handle._impl.parent_container_id
        general_tab_handle: viser.GuiTabHandle = self._server.gui._container_handle_from_uuid[general_tab_id]
        tabs: viser.GuiTabGroupHandle = general_tab_handle._parent

        self._campose_controller = CamPoseController(self, self._server, tabs)

        if block:
            while True:
                time.sleep(999)
