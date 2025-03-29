import os
import os.path as osp
from argparse import Namespace
from random import gauss

import numpy as np
import pandas as pd
import torch
from PIL import Image
from plyfile import PlyData

from internal.dataparsers import DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap
from internal.dataset import CacheDataLoader, Dataset
from internal.metrics.vanilla_metrics import VanillaMetricsImpl
from myimpl.metrics.scaffold_metrics import ScaffoldMetrics
from myimpl.models.scaffold_gaussian import ScaffoldLoDGaussian
from myimpl.renderers.scaffold_renderer import ScaffoldLoDRenderer

dataset_path = "datasets/Mip-NeRF360/v2/bicycle"
ckpt_path = "tmp/checkpoints/scaffold-ckpt/point_cloud/iteration_60000"
device = torch.device("cuda")
bg_color = torch.zeros((3,), dtype=torch.float32, device=device)
dst_path = "tmp/evaluations1"


def load_data():
    dataparser = Colmap(
        split_mode="experiment",
        eval_image_select_mode="step",
        eval_step=8,
        down_sample_factor=2,
    ).instantiate(dataset_path, os.getcwd(), 0)
    dataparser_outputs = dataparser.get_outputs()
    return dataparser_outputs


def load_model(dataparser_outputs: DataParserOutputs):
    ply_path = osp.join(ckpt_path, "point_cloud.ply")
    property_dict = load_anchor_ply(ply_path)
    color_mlp = torch.jit.load(osp.join(ckpt_path, "color_mlp.pt"), map_location="cpu").state_dict()
    cov_mlp = torch.jit.load(osp.join(ckpt_path, "cov_mlp.pt"), map_location="cpu").state_dict()
    opacity_mlp = torch.jit.load(osp.join(ckpt_path, "opacity_mlp.pt"), map_location="cpu").state_dict()
    state_dict = {}
    for k, v in property_dict.items():
        if v is None:
            continue
        state_dict["gaussians." + k] = torch.from_numpy(v)
        if k == "levels":
            state_dict["gaussians." + k] = state_dict["gaussians." + k].squeeze()
    for name, sd in zip(["opacity", "cov", "color"], [opacity_mlp, cov_mlp, color_mlp]):
        for k, v in sd.items():
            state_dict["gaussian_mlps." + name + "." + k] = v

    # orig scaffold gs
    n_anchors = property_dict["means"].shape[0]
    property_dict["levels"] = torch.zeros((n_anchors,))
    property_dict["extra_levels"] = torch.zeros((n_anchors,))

    octree_params = {
        "_max_level": torch.tensor(0),
        "_optimize_from_level": torch.tensor(0),
        "_standard_dist": torch.tensor(0),
        "_voxel_size": torch.tensor(0),
        "_grid_origin": torch.zeros((3,)),
        "_visibility_threshold": torch.tensor(0),
    }

    # octree_params = torch.load(osp.join(ckpt_path, "octree_params.pt"), map_location="cpu")
    # for k in ["_max_level", "_optimize_from_level", "_visibility_threshold"]:
    #     octree_params[k] = torch.tensor(octree_params[k])
    state_dict.update(octree_params)

    gaussian_model = ScaffoldLoDGaussian().instantiate()
    # gaussian_model.setup_from_pcd(
    #     dataparser_outputs.point_cloud.xyz, dataparser_outputs.point_cloud.rgb, dataparser_outputs.train_set.cameras
    # )
    gaussian_model.setup_from_number(property_dict["means"].shape[0])
    # points = torch.from_numpy(dataparser_outputs.point_cloud.xyz).float()
    # gaussian_model.set_octree_properties(points, dataparser_outputs.train_set.cameras)

    gaussian_model.load_state_dict(state_dict, strict=False)
    gaussian_model.to(device)
    return gaussian_model


def load_renderer():
    renderer = ScaffoldLoDRenderer().instantiate()
    renderer.setup(stage="validation")
    renderer = renderer.to(device)
    return renderer


def load_metrics():
    metrics = ScaffoldMetrics().instantiate()
    metrics.setup(stage="validation", pl_module=None)
    metrics = metrics.to(device)
    for k, v in metrics.no_state_dict_models.items():
        metrics.no_state_dict_models[k] = v.to(device)
    return metrics


def load_anchor_ply(path):
    plydata = PlyData.read(path)
    infos = plydata.obj_info
    var_dict = {}
    for info in infos:
        var_name = info.split(" ")[0]
        var_dict[var_name] = float(info.split(" ")[1])

    anchor = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    ).astype(np.float32)

    levels, extra_levels, scales = [None] * 3
    if "level" in plydata.elements[0]:
        levels = np.asarray(plydata.elements[0]["level"])[..., np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"]).astype(np.float32)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((anchor.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((anchor.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

    # anchor_feat
    anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
    anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split("_")[-1]))
    anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
    for idx, attr_name in enumerate(anchor_feat_names):
        anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

    offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
    offset_names = sorted(offset_names, key=lambda x: int(x.split("_")[-1]))
    offsets = np.zeros((anchor.shape[0], len(offset_names)))
    for idx, attr_name in enumerate(offset_names):
        offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
    offsets = offsets.reshape((offsets.shape[0], 3, -1)).transpose((0, 2, 1))

    return {
        "means": anchor,
        "offsets": offsets,
        "scales": scales,
        "levels": levels,
        "extra_levels": extra_levels,
        # "rotations": rots,
        "anchor_features": anchor_feats,
    }


def render(dataparser_outputs, gaussian_model, renderer, metrics):
    dataloader = CacheDataLoader(
        Dataset(dataparser_outputs.test_set, camera_device=device, image_device=device),
        max_cache_num=64,
        shuffle=False,
    )

    os.makedirs(osp.join(dst_path, "images"), exist_ok=True)
    _pl_module = Namespace(trainer=Namespace(global_step=1 << 30))
    metric_dicts = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            camera, image_info, _ = batch
            output_pkg = renderer(camera, gaussian_model, bg_color)
            _metric_dict, _ = metrics.get_validate_metrics(_pl_module, gaussian_model, batch, output_pkg)

            img = (output_pkg["render"].clamp(0.0, 1.0) * 255).byte().permute(1, 2, 0).detach().cpu().numpy()
            img = Image.fromarray(img, "RGB")
            img.save(osp.join(dst_path, "images", "{:05d}.png".format(idx)))
            metric_dicts.append(
                {
                    "idx": "{:05d}.png".format(idx),
                    "img_name": image_info[0],
                    "loss": _metric_dict["loss"].item(),
                    "rgb_diff": _metric_dict["rgb_diff"].item(),
                    "ssim": _metric_dict["ssim"].item(),
                    "psnr": _metric_dict["psnr"].item(),
                    "lpips": _metric_dict["lpips"].item(),
                }
            )
    df = pd.DataFrame(metric_dicts)
    df.to_csv(osp.join(dst_path, "metrics.csv"), index=False)


if __name__ == "__main__":
    # dataparser_outputs = load_data()
    dataparser_outputs = None
    gaussian_model = load_model(dataparser_outputs)
    renderer = load_renderer()
    metrics = load_metrics()

    # render(dataparser_outputs, gaussian_model, renderer, metrics)

    ckpt = torch.load("tmp/checkpoints/gspl_iteration1.ckpt", map_location="cpu")
    ckpt["state_dict"].update({"gaussian_model." + k: v for k, v in gaussian_model.state_dict().items()})
    del ckpt["state_dict"]["gaussian_model._camera_infos"]
    del ckpt["state_dict"]["gaussian_model.gaussians.rotations"]
    torch.save(ckpt, "tmp/checkpoints/edited.ckpt")
