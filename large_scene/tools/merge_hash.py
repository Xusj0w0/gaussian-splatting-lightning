import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap
from internal.density_controllers.vanilla_density_controller import \
    VanillaDensityController
from internal.models.appearance_feature_gaussian import \
    AppearanceFeatureGaussianModel
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.models.vanilla_gaussian import VanillaGaussian
from internal.renderers.gsplat_mip_splatting_renderer_v2 import \
    GSplatMipSplattingRendererV2
from internal.renderers.gsplat_v1_renderer import GSplatV1Renderer
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from large_scene.utils.partition import Partition
from large_scene.utils.partition_training import (PartitionTraining,
                                                  PartitionTrainingConfig)
from myimpl.models.grid_gaussians import ScaffoldGaussianModelMixin


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("partition_dir")
    parser.add_argument("--project", "-p", type=str, required=False, help="Project Name")
    parser.add_argument("--output_path", "-o", type=str, required=False)
    parser.add_argument("--min-images", type=int, default=32)
    # ===== for evaluation ======
    parser.add_argument("--retain-appearance", action="store_true", default=False)
    parser.add_argument("--left-optimized", action="store_true", default=False)
    parser.add_argument("--right-optimized", action="store_true", default=False)
    # ===== for evaluation ======
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()

    if args.output_path is None:
        # args.output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", args.project, "merged.ckpt")
        args.output_path = os.path.join(os.getcwd(), "outputs", args.project, "merged/merged.ckpt")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    elif not args.output_path.endswith(".ckpt"):
        args.output_path += ".ckpt"

    if not args.preprocess:
        assert os.path.exists(args.output_path) is False, "output file '{}' already exists".format(args.output_path)

    return args


def update_ckpt(
    ckpt, merged_gaussians, max_sh_degree, retain_appearance: bool, scaffold_infos: Optional[Dict[str, Any]] = None
):
    if retain_appearance:
        from internal.models.appearance_feature_gaussian import \
            AppearanceFeatureGaussian

        ckpt["hyper_parameters"]["gaussian"] = AppearanceFeatureGaussian(
            sh_degree=max_sh_degree,
            appearance_feature_dims=ckpt["hyper_parameters"]["gaussian"].appearance_feature_dims,
        )
    # support partition rendering of scaffold based model
    elif scaffold_infos is not None:
        from myimpl.models.hash_grid_assisted_gaussian import \
            HashGridAssistedGaussianModel
        from myimpl.models.partitionable_implicit_grid_gaussian import (
            PartitionableImplicitGridGaussian,
            PartitionableImplicitLoDGridGaussian)

        anchor_partition_ids = torch.cat(scaffold_infos["anchor_partition_ids"], 0)
        ckpt["state_dict"]["gaussian_model._anchor_partition_ids"] = anchor_partition_ids

        # remove original gaussian_mlps states
        keys_removed = []
        for k in ckpt["state_dict"]:
            if k.startswith("gaussian_model.gaussian_mlps"):
                keys_removed.append(k)
        for k in keys_removed:
            del ckpt["state_dict"][k]
        # add new states
        ckpt["state_dict"].update(
            {
                f"gaussian_model.gaussian_mlps.{k}": v
                for k, v in nn.ModuleDict(scaffold_infos["mlps"]).state_dict().items()
            }
        )

        if ckpt["state_dict"].get("gaussian_model.gaussian_mlps.hash_grid_feature", None) is not None:
            model = ckpt["hyper_parameters"]["gaussian"].instantiate()
            model.load_state_dict(ckpt["state_dict"])

        # modify hyperparameter
        orig_gaussian = ckpt["hyper_parameters"]["gaussian"]
        if ckpt["state_dict"].get("gaussian_model.gaussians.levels", None) is not None: 
            gaussian_params = {
                k: getattr(orig_gaussian, k)
                for k, v in PartitionableImplicitLoDGridGaussian.__dataclass_fields__.items()
                if k in orig_gaussian.__dict__ and v.init
            }
            gaussian_params.update({"feature_dim": orig_gaussian.mlp_in_features})
            gaussian_params.update({"partition_ids": torch.unique(anchor_partition_ids).tolist()})
            ckpt["hyper_parameters"]["gaussian"] = PartitionableImplicitLoDGridGaussian(**gaussian_params)
        else:
            gaussian_params = {
                k: getattr(orig_gaussian, k)
                for k in PartitionableImplicitGridGaussian.__dataclass_fields__
                if k in orig_gaussian.__dict__
            }
            gaussian_params.update({"partition_ids": torch.unique(anchor_partition_ids).tolist()})
            ckpt["hyper_parameters"]["gaussian"] = PartitionableImplicitGridGaussian(**gaussian_params)
    else:
        # replace `AppearanceFeatureGaussian` with `VanillaGaussian`
        ckpt["hyper_parameters"]["gaussian"] = VanillaGaussian(sh_degree=max_sh_degree)

        # remove `GSplatAppearanceEmbeddingRenderer`'s states from ckpt
        state_dict_key_to_delete = []
        for i in ckpt["state_dict"]:
            if i.startswith("renderer."):
                state_dict_key_to_delete.append(i)
        for i in state_dict_key_to_delete:
            del ckpt["state_dict"][i]

    # replace `GSplatAppearanceEmbeddingRenderer` with `GSPlatRenderer`
    anti_aliased = True
    kernel_size = 0.3
    if isinstance(ckpt["hyper_parameters"]["renderer"], VanillaRenderer):
        anti_aliased = False
    elif (
        isinstance(ckpt["hyper_parameters"]["renderer"], GSplatMipSplattingRendererV2)
        or ckpt["hyper_parameters"]["renderer"].__class__.__name__ == "GSplatAppearanceEmbeddingMipRenderer"
    ):
        kernel_size = ckpt["hyper_parameters"]["renderer"].filter_2d_kernel_size

    if retain_appearance:
        from internal.renderers.gsplat_appearance_embedding_renderer import \
            GSplatAppearanceEmbeddingRenderer

        ckpt["hyper_parameters"]["renderer"] = GSplatAppearanceEmbeddingRenderer(
            anti_aliased=anti_aliased,
            filter_2d_kernel_size=kernel_size,
            separate_sh=True,
            tile_based_culling=getattr(ckpt["hyper_parameters"]["renderer"], "tile_based_culling", False),
            model=ckpt["hyper_parameters"]["renderer"].model,
        )
    elif scaffold_infos is not None:
        from myimpl.renderers.grid_renderer import GridGaussianRenderer
        orig_renderer = ckpt["hyper_parameters"]["renderer"]
        ckpt["hyper_parameters"]["renderer"] = GridGaussianRenderer(
            **{k: getattr(orig_renderer, k) for k, v in GridGaussianRenderer.__dataclass_fields__.items() if v.init}
        )
    else:
        ckpt["hyper_parameters"]["renderer"] = GSplatV1Renderer(
            anti_aliased=anti_aliased,
            filter_2d_kernel_size=kernel_size,
            separate_sh=getattr(ckpt["hyper_parameters"]["renderer"], "separate_sh", True),
            tile_based_culling=getattr(ckpt["hyper_parameters"]["renderer"], "tile_based_culling", False),
        )

    # remove existing Gaussians from ckpt
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith("gaussian_model.gaussians.") or i.startswith("frozen_gaussians."):
            del ckpt["state_dict"][i]

    # remove optimizer states
    ckpt["optimizer_states"] = []

    # reinitialize density controller states
    if isinstance(ckpt["hyper_parameters"]["density"], VanillaDensityController):
        for k in list(ckpt["state_dict"].keys()):
            if k.startswith("density_controller."):
                ckpt["state_dict"][k] = torch.zeros(
                    (merged_gaussians["means"].shape[0], *ckpt["state_dict"][k].shape[1:]),
                    dtype=ckpt["state_dict"][k].dtype,
                )

    # add merged gaussians to ckpt
    for k, v in merged_gaussians.items():
        ckpt["state_dict"]["gaussian_model.gaussians.{}".format(k)] = v


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

    MERGABLE_PROPERTY_NAMES = [
        "means",
        "shs_dc",
        "shs_rest",
        "scales",
        "rotations",
        "offsets",
        "anchor_features",
        "levels",
        "extra_levels",
    ]

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
    mergable_partitions, orientation_transformation = partition_training.get_trained_partitions()
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
            t.set_description("{}".format(partition_id_str))

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
                t.write(
                    "[NOTE]bounding box of {} updated to {}".format(
                        partition_training.partition_coordinates.id[partition_idx].tolist(),
                        bounding_box,
                    )
                )

            gaussian_model, _, _ = partition_training.split_partition_gaussians(
                ckpt,
                bounding_box,
            )
            gaussian_model.to("cuda")

            if getattr(gaussian_model, "gaussian_mlps", None) is not None:
                # if not args.preprocess:
                if scaffold_infos is None:
                    scaffold_infos = {"mlps": {}, "anchor_partition_ids": []}
                anchors = gaussian_model.get_anchors
                mask = anchors.new_ones((anchors.shape[0],), dtype=torch.bool)
                features = gaussian_model.compute_anchor_features(anchors, mask)
                gaussian_model.set_property("anchor_features", features)
                scaffold_infos["anchor_partition_ids"].append(
                    torch.full((gaussian_model.get_means().shape[0],), partition_idx, dtype=torch.int)
                )
                for k in ["opacity", "cov", "color"]:
                    if scaffold_infos["mlps"].get(k, None) is None:
                        scaffold_infos["mlps"][k] = nn.ModuleDict()
                    scaffold_infos["mlps"][k][str(partition_idx)] = getattr(gaussian_model, f"get_{k}_mlp")

            if args.preprocess:
                update_ckpt(
                    ckpt,
                    {
                        k: gaussian_model.get_property(k)
                        for k in MERGABLE_PROPERTY_NAMES
                        if k in gaussian_model.property_names
                    },
                    gaussian_model.max_sh_degree,
                    retain_appearance=args.retain_appearance,
                    scaffold_infos=scaffold_infos,
                )

                output_filename_suffix = ""
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

    update_ckpt(
        ckpt,
        merged_gaussians,
        gaussian_model.max_sh_degree,
        retain_appearance=args.retain_appearance,
        scaffold_infos=scaffold_infos,
    )

    # save
    print("Saving...")
    torch.save(ckpt, args.output_path)
    print("Saved to '{}'".format(args.output_path))

    viewer_args = ["python", "viewer.py", args.output_path]
    if orientation_transformation is not None:
        viewer_args += ["--up"]
        viewer_args += ["{:.4f}".format(i) for i in (partition_training.scene["extra_data"]["up"]).tolist()]
    print("The command to start web viewer:" " {}".format(" ".join(viewer_args)))


if __name__ == "__main__":
    main()
