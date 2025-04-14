import math
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors

from myimpl.models.grid_gaussians import (GridGaussianModel,
                                          LoDGridGaussianModel,
                                          ScaffoldGaussianModelMixin)


class GridGaussianType:
    Implicit: int = 1
    ImplicitLoD: int = 2
    Explicit: int = 3
    ExplicitLoD: int = 4


class GridGaussianUtils:
    # fmt: off
    GRID_GAUSSIAN_PROPERTIES = {
        1: {
            "buffers": ["voxel_size", "grid_origin"],
            "properties": ["anchors", "offsets", "scales", "rotations",
                        "anchor_features",],
            "mlps": ["opacity_mlp", "cov_mlp", "color_mlp", "feature_bank_mlp"],
        },
        2: {
            "buffers": ["voxel_size", "grid_origin", "max_level", "start_level", "standard_dist", "visibility_threshold"],
            "properties": ["anchors", "offsets", "scales", "rotations", "levels", "extra_levels",
                        "anchor_features",],
            "mlps": ["opacity_mlp", "cov_mlp", "color_mlp", "feature_bank_mlp"],
        }
    }
    # fmt: on

    PROPERTY_NAME_ATTR_MAPPING = {
        "anchors": "means",
    }

    MLP_NAME_ATTR_MAPPING = {
        "opacity_mlp": "opacity",
        "cov_mlp": "cov",
        "color_mlp": "color",
        "feature_bank_mlp": "feature_bank",
    }

    @classmethod
    def get_type(cls, model: GridGaussianModel) -> int:
        if not isinstance(model, (GridGaussianModel, LoDGridGaussianModel)):
            raise ValueError('`model` should be ["Implicit", "ImplicitLoD", "Explicit", "ExplicitLoD"]')
        if isinstance(model, LoDGridGaussianModel):
            if isinstance(model, ScaffoldGaussianModelMixin):
                return GridGaussianType.ImplicitLoD
            # elif
            else:
                raise ValueError
        else:
            if isinstance(model, ScaffoldGaussianModelMixin):
                return GridGaussianType.Implicit
            # elif
            else:
                raise ValueError

    @classmethod
    def tensors_from_model(cls, model: GridGaussianModel):
        pt = {}
        type = cls.get_type(model)

        # use `get_xxx` to get buffers
        for buffer in cls.GRID_GAUSSIAN_PROPERTIES[type]["buffers"]:
            attr = getattr(model, buffer)
            pt.update({f"_{buffer}": torch.tensor(attr) if not isinstance(attr, torch.Tensor) else attr})

        # get properties from gaussians container
        for property in cls.GRID_GAUSSIAN_PROPERTIES[type]["properties"]:
            pt.update({property: model.gaussians.get(cls.PROPERTY_NAME_ATTR_MAPPING.get(property, property), None)})

        # get properties from gaussian_mlps container
        for mlp in cls.GRID_GAUSSIAN_PROPERTIES[type]["mlps"]:
            key = cls.MLP_NAME_ATTR_MAPPING.get(mlp, mlp)
            if key in model.gaussian_mlps:
                pt.update({mlp: model.gaussian_mlps[key].state_dict()})

        return pt

    @classmethod
    def buffers_from_ply(cls, plydata: PlyData):
        # load buffers
        infos = {}
        for info in plydata.obj_info:
            key, val = info.strip().split(" ", 1)
            if "," in val:
                infos[key] = [float(v) for v in val.split(",")]
            else:
                infos[key] = float(val)

        buffers = {
            buffer_name: torch.tensor(infos[key], dtype=dtype)
            for buffer_name, (key, dtype) in [
                ["voxel_size", ("voxel_size", torch.float32)],
                ["grid_origin", ("init_pos", torch.float32)],
                ["max_level", ("levels", torch.int)],
                ["start_level", ("init_level", torch.int)],
                ["standard_dist", ("standard_dist", torch.float32)],
                ["visibility_threshold", ("visible_threshold", torch.float32)],
            ]
            if key in infos
        }
        return buffers

    @classmethod
    def properties_from_ply(cls, plydata: PlyData):
        anchors = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        ).astype(np.float32)

        levels, extra_levels = None, None
        if "level" in plydata.elements[0]:
            levels = np.asarray(plydata.elements[0]["level"])[..., np.newaxis].astype(np.int16)
            extra_levels = np.asarray(plydata.elements[0]["extra_level"]).astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((anchors.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((anchors.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split("_")[-1]))
        anchor_feats = np.zeros((anchors.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key=lambda x: int(x.split("_")[-1]))
        offsets = np.zeros((anchors.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1)).transpose((0, 2, 1))

        states = {
            "means": torch.from_numpy(anchors).float(),
            "offsets": torch.from_numpy(offsets).float(),
            "scales": torch.from_numpy(scales).float(),
            "anchor_features": torch.from_numpy(anchor_feats).float(),
        }
        if levels is not None:
            states.update(
                {
                    "levels": torch.from_numpy(levels).squeeze().int(),
                    "extra_levels": torch.from_numpy(extra_levels).int(),
                }
            )

        return states

    @classmethod
    def mlp_from_pt(cls, ckpt_path: str):
        assert all([osp.exists(osp.join(ckpt_path, f"{f}_mlp.pt")) for f in ["color", "cov", "opacity"]])

        states = {}
        for key in ["color", "cov", "opacity"]:
            mlp_path = osp.join(ckpt_path, f"{key}_mlp.pt")
            state_dict = torch.jit.load(mlp_path, map_location="cpu").state_dict()
            for k, v in state_dict.items():
                states[f"{key}.{k}"] = v
        return states

    @classmethod
    def convert_orig_to_ckpt(cls, ckpt_path: str, gspl_ckpt):
        assert osp.exists(osp.join(ckpt_path, "point_cloud.ply"))
        ply_path = osp.join(ckpt_path, "point_cloud.ply")
        plydata = PlyData.read(ply_path)

        buffers = cls.buffers_from_ply(plydata)
        buffers = {f"_{k}": v for k, v in buffers.items()}

        properties = cls.properties_from_ply(plydata)
        properties = {f"gaussian.{k}": v for k, v in properties.items()}

        mlps = cls.mlp_from_pt(ckpt_path)
        mlps = {f"gaussian_mlps.{k}": v for k, v in mlps.items()}

        n_anchors = len(properties["gaussians.means"])
        model = gspl_ckpt["hyper_parameters"]["gaussian"].instantiate()
        model.setup_from_number(n_anchors)

        state_dict = {}
        state_dict.update(buffers)
        state_dict.update(properties)
        state_dict.update(mlps)
        gspl_ckpt["state_dict"].update({"gaussian_model." + k: v for k, v in state_dict.items()})

        return gspl_ckpt
