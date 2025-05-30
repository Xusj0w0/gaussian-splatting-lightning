# fmt: off
# isort: skip_file
# same as utils/merge_partition_v2.py
# modify trained_partition_utils, change import path of utils.fuse_appearance_embeddings_into_shs_dc

import os
import sys
import gc
import json
import argparse
from tqdm.auto import tqdm
from typing import Dict, Optional, Any
from dataclasses import asdict
import torch
import torch.nn as nn
from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.vanilla_gaussian import VanillaGaussian
from internal.models.appearance_feature_gaussian import AppearanceFeatureGaussianModel
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.renderers.gsplat_v1_renderer import GSplatV1Renderer
from internal.renderers.gsplat_mip_splatting_renderer_v2 import GSplatMipSplattingRendererV2
from internal.density_controllers.vanilla_density_controller import VanillaDensityController
from internal.utils.gaussian_model_loader import GaussianModelLoader

from myimpl.models.grid_gaussians import ScaffoldGaussianModelMixin
from large_scene.utils.partition import Partition
from large_scene.utils.partition_training import PartitionTraining, PartitionTrainingConfig
from large_scene.tools.merge import fuse_appearance_features, fuse_mip_filters, update_ckpt, convert_to_embedding_optimized_ckpt_file_path


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("partition_dir")
    parser.add_argument("--project", "-p", type=str, required=False,
                        help="Project Name")
    parser.add_argument("--partition_id", type=str, default="all")
    parser.add_argument("--min-images", type=int, default=32)
    # ===== for evaluation ======
    parser.add_argument("--retain-appearance", action="store_true", default=False)
    parser.add_argument("--left-optimized", action="store_true", default=False)
    parser.add_argument("--right-optimized", action="store_true", default=False)
    # ===== for evaluation ======
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()

    args.preprocess = True

    if not args.preprocess:
        assert os.path.exists(args.output_path) is False, "output file '{}' already exists".format(args.output_path)

    return args


def get_trained_partitions(self: PartitionTraining):
    trainable_partition_idx_list = self.get_trainable_partition_idx_list(
        min_images=self.config.min_images,
        n_processes=self.config.n_processes,
        process_id=self.config.process_id,
    )
    partition_bounding_boxes = self.partition_coordinates.get_bounding_boxes()

    trained_partitions = []
    for partition_idx in tqdm(trainable_partition_idx_list, desc="Searching checkpoints"):
        partition_id_str = self.get_partition_id_str(partition_idx)
        trained_file = os.path.join(
            self.project_output_dir,
            self.get_partition_trained_step_filename(partition_idx),
        )
        if not os.path.exists(trained_file):
            continue
        model_dir = os.path.join(self.project_output_dir, partition_id_str)
        ckpt_file = GaussianModelLoader.search_load_file(model_dir)
        if not ckpt_file.endswith(".ckpt"):
            continue
        trained_partitions.append((
            partition_idx,
            partition_id_str,
            ckpt_file,
            partition_bounding_boxes[partition_idx],
        ))

    orientation_transformation = (
        self.scene["extra_data"]["rotation_transform"]
        if self.scene["extra_data"] is not None
        else None
    )
    return trained_partitions, orientation_transformation


def main():
    """
    Overall pipeline:
        * Load the partition data
        * Get trainable partitions and their checkpoint filenames
        * For each partition
          * Load the checkpoint
          * Extract Gaussians falling into the partition bounding box
          * Fuse appearance features into SHs
        * Merge all extracted Gaussians
        * Update the checkpoint
          * Replace GaussianModel with the vanilla one
          * Replace `AppearanceEmbeddingRenderer` with the `GSPlatRenderer`
          * Clear optimizers' states
          * Re-initialize density controller's states
          * Replace with merged Gaussians
        * Saving
    """

    MERGABLE_PROPERTY_NAMES = ["means", "shs_dc", "shs_rest", "scales", "rotations", "offsets",
                               "anchor_features", "levels", "extra_levels"]

    args = parse_args()

    if args.retain_appearance:
        assert args.preprocess
    assert int(args.left_optimized) + int(args.right_optimized) != 2

    if args.retain_appearance:
        MERGABLE_PROPERTY_NAMES.append(AppearanceFeatureGaussianModel._appearance_feature_name)

    torch.autograd.set_grad_enabled(False)

    partition_training = PartitionTraining(
        PartitionTrainingConfig(
            project_name=args.project,
            min_images=args.min_images,
            n_processes=1,
            process_id=1,
            dry_run=False,
            extra_epoches=0,
        )
    )
    mergable_partitions, orientation_transformation = get_trained_partitions(partition_training)

    image_name_to_camera = None

    gaussians_to_merge = {}

    partition_bounding_boxes = partition_training.partition_coordinates.get_bounding_boxes()
    scene_bounding_box = (
        torch.min(partition_bounding_boxes.min, dim=0).values,
        torch.max(partition_bounding_boxes.max, dim=0).values,
    )

    def isclose(a, b):
        return torch.isclose(a, b, atol=1e-4)

    scaffold_infos = None
    with tqdm(mergable_partitions, desc="Pre-processing") as t:
        for partition_idx, partition_id_str, ckpt_file, bounding_box in t:
            if args.partition_id != "all" and partition_id_str != args.partition_id:
                continue

            t.set_description("{}".format(partition_id_str))

            # ===== for evaluation =====
            if args.left_optimized:
                ckpt_file = convert_to_embedding_optimized_ckpt_file_path(ckpt_file, "left")
            elif args.right_optimized:
                ckpt_file = convert_to_embedding_optimized_ckpt_file_path(ckpt_file, "right")
            # ===== for evaluation =====

            t.write("Loading {}".format(ckpt_file))

            ckpt = torch.load(ckpt_file, map_location="cpu")

            t.write("Splitting...")
            # include background if the partition locates at the border
            # TODO: deal with the non-rectangular case
            bounding_box_updated = False
            if isclose(bounding_box.min[0], scene_bounding_box[0][0]):
                # x == scene bbox x min -> bbox.x_min = -inf
                bounding_box.min[0] = -torch.inf
                bounding_box_updated = True
            if isclose(bounding_box.min[1], scene_bounding_box[0][1]):
                # y == scene bbox y min -> bbox.y_min = -inf
                bounding_box.min[1] = -torch.inf
                bounding_box_updated = True
            if isclose(bounding_box.max[0], scene_bounding_box[1][0]):
                # x == scene bbox x max -> bbox.x_max = inf
                bounding_box.max[0] = torch.inf
                bounding_box_updated = True
            if isclose(bounding_box.max[1], scene_bounding_box[1][1]):
                # x == scene bbox x max -> bbox.x_max = inf
                bounding_box.max[1] = torch.inf
                bounding_box_updated = True

            if bounding_box_updated:
                t.write("[NOTE]bounding box of {} updated to {}".format(
                    partition_training.partition_coordinates.id[partition_idx].tolist(),
                    bounding_box,
                ))

            gaussian_model, _, _ = partition_training.split_partition_gaussians(
                ckpt,
                bounding_box,
            )

            if isinstance(gaussian_model, AppearanceFeatureGaussianModel) and not args.retain_appearance:
                with open(os.path.join(
                        os.path.dirname(os.path.dirname(ckpt_file)),
                        "cameras.json"
                ), "r") as f:
                    cameras_json = json.load(f)

                # the dataset will only be loaded once
                if image_name_to_camera is None:
                    t.write("Loading colmap model...")

                    dataparser_config = Colmap(
                        split_mode="reconstruction",
                        eval_step=64,
                        points_from="random",
                    )
                    for i in ["image_dir", "mask_dir", "scene_scale", "reorient", "appearance_groups", "down_sample_factor", "down_sample_rounding_mode"]:
                        setattr(dataparser_config, i, getattr(ckpt["datamodule_hyper_parameters"]["parser"], i))
                    dataparser_outputs = dataparser_config.instantiate(
                        path=partition_training.dataset_path,
                        output_path=os.getcwd(),
                        global_rank=0,
                    ).get_outputs()

                    image_name_to_camera = {}
                    for idx in range(len(dataparser_outputs.train_set)):
                        image_name = dataparser_outputs.train_set.image_names[idx]
                        camera = dataparser_outputs.train_set.cameras[idx]
                        image_name_to_camera[image_name] = camera

                t.write("Fusing appearance features...")
                fuse_appearance_features(
                    ckpt,
                    gaussian_model,
                    cameras_json,
                    image_name_to_camera=image_name_to_camera,
                )

            if isinstance(gaussian_model, MipSplattingModelMixin):
                t.write("Fusing MipSplatting filters...")
                fuse_mip_filters(gaussian_model)

            if getattr(gaussian_model, "gaussian_mlps", None) is not None:
                # if not args.preprocess:
                if scaffold_infos is None:
                    scaffold_infos = {"mlps": {}, "anchor_partition_ids": []}
                scaffold_infos["anchor_partition_ids"].append(torch.full((gaussian_model.get_means().shape[0],), partition_idx, dtype=torch.int))
                for k in ["opacity", "cov", "color"]:
                    if scaffold_infos["mlps"].get(k, None) is None:
                        scaffold_infos["mlps"][k] = nn.ModuleDict()
                    scaffold_infos["mlps"][k][str(partition_idx)] = getattr(gaussian_model, f"get_{k}_mlp")

            if args.preprocess:
                update_ckpt(ckpt, {k: gaussian_model.get_property(k) for k in MERGABLE_PROPERTY_NAMES if k in gaussian_model.property_names}, gaussian_model.max_sh_degree, retain_appearance=args.retain_appearance, scaffold_infos=scaffold_infos)

                output_filename_suffix = ""
                if args.retain_appearance:
                    output_filename_suffix += "-retain_appearance"
                if args.left_optimized:
                    output_filename_suffix += "-left_optimized"
                elif args.right_optimized:
                    output_filename_suffix += "-right_optimized"

                output_filename = os.path.join(
                    os.path.dirname(os.path.dirname(ckpt_file)),
                    "preprocessed{}.ckpt".format(output_filename_suffix),
                )
                torch.save(ckpt, output_filename)
                t.write("Saved to {}".format(output_filename))
            else:
                for i in MERGABLE_PROPERTY_NAMES:
                    if i in gaussian_model.property_names:
                        gaussians_to_merge.setdefault(i, []).append(gaussian_model.get_property(i))

    if args.preprocess:
        return

    # merge
    print("Merging...")
    merged_gaussians = {}
    for k, v in gaussians_to_merge.items():
        merged_gaussians[k] = torch.concat(v, dim=0)
        # release merged one to avoid OOM
        v.clear()
        gc.collect()
        torch.cuda.empty_cache()

    update_ckpt(ckpt, merged_gaussians, gaussian_model.max_sh_degree, retain_appearance=args.retain_appearance, scaffold_infos=scaffold_infos)

    # save
    print("Saving...")
    torch.save(ckpt, args.output_path)
    print("Saved to '{}'".format(args.output_path))

    viewer_args = ["python", "viewer.py", args.output_path]
    if orientation_transformation is not None:
        viewer_args += ["--up"]
        viewer_args += ["{:.4f}".format(i) for i in (partition_training.scene["extra_data"]["up"]).tolist()]
    print("The command to start web viewer:"
          " {}".format(" ".join(viewer_args)))


def test_main():
    sys.argv = [
        __file__,
        os.path.expanduser("~/dataset/JNUCar_undistorted/colmap/drone/dense_max_2048/0/partitions-size_3.0-enlarge_0.1-visibility_0.9_0.1"),
        "-p", "JNUAerial-0820",
    ]
    main()


if __name__ == "__main__":
    main()
