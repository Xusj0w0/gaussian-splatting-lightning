# modified from notebook/meganerf_rubble_split.ipynb
# replace reorient with manually-specified manhattan transformation
import os
import os.path as osp
from argparse import ArgumentParser, Namespace
from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import ConfigDict
from tqdm.auto import tqdm

import internal.utils.colmap as colmap_utils
from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.utils.partitioning_utils import PartitionableScene, SceneConfig

torch.set_grad_enabled(False)
torch.set_printoptions(precision=16)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to dataset. Containing directories images, sparse, etc.",
    )
    parser.add_argument("--manhattan_trans", type=str, default="", help="Relative path to dataset_path")
    parser.add_argument("--origin", type=str, default="0,0", help="Scene origin. Like '0.0,0.0'")
    parser.add_argument(
        "--partition_size", type=str, default="100", help="Scene partition size. Like '60.0' or '60.0,60.0'"
    )
    parser.add_argument(
        "--location_based_enlarge", type=float, default=0.1, help="Enlarge the bbox defined by camera potitions."
    )
    parser.add_argument("--visibility_based_distance", type=float, default=0.9, help="")
    parser.add_argument("--visibility_threshold", type=float, default=1.0 / 6.0)
    parser.add_argument("--bounding_box_based_visibility", type=bool, default=False)
    parser.add_argument("--min_cameras", type=int, default=3)
    parser.add_argument("--max_errors", type=float, default=2.0)
    return parser


def parse_manhattan_trans(dataset_path: str, manhattan_trans: str):
    if len(manhattan_trans) > 0:
        manhattan_path = osp.join(dataset_path, manhattan_trans)
        assert osp.exists(manhattan_path), "File to manhattan transformation not found."
        try:
            with open(manhattan_path, "r") as f:
                trans_mat = " ".join([l.strip() for l in f.readlines()])
            seq = [float(s) for s in trans_mat.split()]
        except:
            raise ValueError("Parse manhattan transformation failed.")
        assert len(seq) == 16, "Invalid manhattan transformation."
        manhattan_trans = np.array(seq).reshape(4, 4)
    else:
        manhattan_trans = np.eye(4)
    return manhattan_trans


def load_colmap_model(dataset_path):
    colmap_model = colmap_utils.read_model(os.path.join(dataset_path, "sparse"))
    colmap_model = {
        "cameras": colmap_model[0],
        "images": colmap_model[1],
        "points3D": colmap_model[2],
    }
    return colmap_model


def parser_cameras(colmap_images):
    R_list = []
    T_list = []
    image_name_list = []
    image_idx_to_key = []

    for idx, key in enumerate(colmap_images):
        extrinsics = colmap_images[key]
        image_name_list.append(extrinsics.name)

        R = torch.tensor(extrinsics.qvec2rotmat(), dtype=torch.float)
        T = torch.tensor(extrinsics.tvec, dtype=torch.float)

        R_list.append(R)
        T_list.append(T)
        image_idx_to_key.append(key)

    R = torch.stack(R_list)
    T = torch.stack(T_list)

    assert image_idx_to_key[0] == list(colmap_images.keys())[0]
    assert image_idx_to_key[-1] == list(colmap_images.keys())[-1]

    R.shape, T.shape, len(image_idx_to_key),

    # calculate camera-to-world transform matrix
    w2c = torch.zeros(size=(R.shape[0], 4, 4), dtype=R.dtype)
    w2c[:, :3, :3] = R
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1.0
    c2w = torch.linalg.inv(w2c)
    return c2w, image_name_list, image_idx_to_key


def parse_points(colmap_points3D):
    max_point_index = max(colmap_points3D.keys())
    point_xyzs = torch.zeros((max_point_index + 1, 3), dtype=torch.float)
    point_rgbs = torch.zeros((max_point_index + 1, 3), dtype=torch.uint8)
    point_errors = torch.ones((max_point_index + 1), dtype=torch.float).fill_(255.0)
    point_n_images = torch.zeros((max_point_index + 1), dtype=torch.int)

    for idx, point in tqdm(colmap_points3D.items()):
        point_xyzs[idx] = torch.from_numpy(point.xyz)
        point_rgbs[idx] = torch.from_numpy(point.rgb)
        point_errors[idx] = torch.from_numpy(point.error)
        point_n_images[idx] = point.image_ids.shape[0]
    return point_xyzs, point_rgbs, point_errors, point_n_images


def partition(dataset_path, config: Namespace, manhattan_trans=None):
    colmap_model = load_colmap_model(dataset_path)
    c2w, image_name_list, image_idx_to_key = parser_cameras(colmap_model["images"])
    point_xyzs, point_rgbs, point_errors, point_n_images = parse_points(colmap_model["points3D"])

    manhattan_trans = parse_manhattan_trans(dataset_path, manhattan_trans)
    manhattan_trans = torch.from_numpy(manhattan_trans).to(point_xyzs[0])
    # extra_trans = torch.tensor(
    #     [
    #         [1, 0, 0, 0],
    #         [0, 0, 1, 0],
    #         [0, -1, 0, 0],
    #         [0, 0, 0, 1],
    #     ]
    # ).to(manhattan_trans)
    # manhattan_trans = extra_trans @ manhattan_trans

    reoriented_camera_centers = c2w[:, :3, 3] @ manhattan_trans[:3, :3].T + manhattan_trans[:3, -1]
    reoriented_point_cloud_xyz = point_xyzs @ manhattan_trans[:3, :3].T + manhattan_trans[:3, -1]
    valid_point_mask = point_n_images > 0
    valid_reoriented_point_xyzs, valid_point_rgbs = (
        reoriented_point_cloud_xyz[valid_point_mask],
        point_rgbs[valid_point_mask],
    )

    scene_config = SceneConfig(
        origin=torch.tensor(config.origin),
        partition_size=config.partition_size,
    )
    scene = PartitionableScene(scene_config, reoriented_camera_centers[..., :2])
    scene.get_bounding_box_by_camera_centers()
    scene.get_scene_bounding_box()
    scene.build_partition_coordinates()

    # some parameters may need to be changed
    scene_config.location_based_enlarge = config.location_based_enlarge
    scene_config.visibility_based_distance = (
        config.visibility_based_distance
    )  # enlarge bounding box by `partition_size * max_visible_distance`, only those cameras inside this enlarged box will be used for visibility based assignment
    scene_config.visibility_threshold = config.visibility_threshold
    # bounding box based visibility
    scene_config.bounding_box_based_visibility = config.bounding_box_based_visibility
    output_path = os.path.join(dataset_path, scene.build_output_dirname())
    os.makedirs(osp.join(output_path, "figs"), exist_ok=True)

    save_plot(scene.plot_scene_bounding_box, osp.join(output_path, "figs", "bounding_boxes.png"))
    save_plot(scene.plot_partitions, osp.join(output_path, "figs", "partitions.png"))

    plot_pcd(
        valid_reoriented_point_xyzs,
        valid_point_rgbs,
        reoriented_camera_centers,
        output_path,
    )

    # location based assignment
    location_based_assignment(scene)

    # visibility based assignment
    # filter out points with large errors
    shared_point_mask = prefilter(point_n_images, point_errors, config.min_cameras, config.max_errors)
    # visibility based assignment
    visibility_based_assignment(
        scene,
        reoriented_point_cloud_xyz,
        shared_point_mask,
        image_idx_to_key,
        colmap_model,
    )

    for idx in range(len(scene.partition_coordinates)):
        save_plot(
            scene.plot_partition_assigned_cameras,
            osp.join(output_path, "figs", f"partition_{idx}.png"),
            partition_idx=idx,
            point_xyzs=valid_reoriented_point_xyzs,
            point_rgbs=valid_point_rgbs,
            point_sparsify=max(valid_reoriented_point_xyzs.shape[0] // 51200, 1),
        )

    save_partitioning_info(scene, c2w, output_path, manhattan_trans, image_name_list)


def save_partitioning_info(scene, c2w, output_path, manhattan_trans, image_name_list):
    torch.load(
        scene.save(
            output_path,
            extra_data={
                "up": torch.linalg.inv(manhattan_trans)[:3, 1],
                "rotation_transform": manhattan_trans,
            },
        )
    )
    is_images_assigned_to_partitions = torch.logical_or(
        scene.is_camera_in_partition, scene.is_partitions_visible_to_cameras
    )
    written_idx_list = []
    for partition_idx in tqdm(list(range(is_images_assigned_to_partitions.shape[0]))):
        partition_image_indices = is_images_assigned_to_partitions[partition_idx].nonzero().squeeze(-1).tolist()
        if len(partition_image_indices) == 0:
            continue
        written_idx_list.append(partition_idx)
        camera_list = []
        with open(
            os.path.join(output_path, "{}.txt".format(scene.partition_coordinates.get_str_id(partition_idx))), "w"
        ) as f:
            for image_index in partition_image_indices:
                f.write(image_name_list[image_index])
                f.write("\n")
                # below camera list is just for visualization, not for training, so its camera intrinsics are fixed values
                color = [0, 0, 255]
                if scene.is_partitions_visible_to_cameras[partition_idx][image_index]:
                    color = [255, 0, 0]
                camera_list.append(
                    {
                        "id": image_index,
                        "img_name": image_name_list[image_index],
                        "width": 1920,
                        "height": 1080,
                        "position": c2w[image_index][:3, 3].numpy().tolist(),
                        "rotation": c2w[image_index][:3, :3].numpy().tolist(),
                        "fx": 1600,
                        "fy": 1600,
                        "color": color,
                    }
                )

        # with open(
        #     os.path.join(
        #         output_path,
        #         f"cameras-{scene.partition_coordinates.get_str_id(partition_idx)}.json",
        #     ),
        #     "w",
        # ) as f:
        #     json.dump(camera_list, f, indent=4, ensure_ascii=False)


def location_based_assignment(scene):
    scene.camera_center_based_partition_assignment()


def prefilter(point_n_images, point_errors, min_cameras=3, max_errors=2.0):
    return torch.logical_and(
        torch.ge(point_n_images, min_cameras),
        torch.le(point_errors, max_errors),
    )


def visibility_based_assignment(
    scene,
    reoriented_point_cloud_xyz,
    shared_point_mask,
    image_idx_to_key,
    colmap_model,
):
    if scene.scene_config.bounding_box_based_visibility:

        def get_image_points(image_idx: int):
            image_key = image_idx_to_key[image_idx]  # the id in colmap sparse model
            # get image size
            camera = colmap_model["cameras"][colmap_model["images"][image_key].camera_id]
            n_pixels = camera.width * camera.height
            # get valid points
            points_xys = torch.from_numpy(colmap_model["images"][image_key].xys)
            points_ids = torch.from_numpy(colmap_model["images"][image_key].point3D_ids)
            valid_mask = points_ids > 0
            points_xys = points_xys[valid_mask]
            points_ids = points_ids[valid_mask]
            # filter
            points_ids *= shared_point_mask[points_ids]
            filter_mask = points_ids > 0
            points_ids = points_ids[filter_mask]
            points_xys = points_xys[filter_mask]
            return points_xys, reoriented_point_cloud_xyz[points_ids], n_pixels

    else:

        def get_image_points(image_idx: int):
            image_key = image_idx_to_key[image_idx]
            # get valid points
            points_ids = torch.from_numpy(colmap_model["images"][image_key].point3D_ids)
            points_ids = points_ids[points_ids > 0]
            # filter
            points_ids *= shared_point_mask[points_ids]
            points_ids = points_ids[points_ids > 0]
            return reoriented_point_cloud_xyz[points_ids]

    scene.calculate_camera_visibilities(
        point_getter=get_image_points,
        device=reoriented_point_cloud_xyz.device,
    )
    scene.visibility_based_partition_assignment()


def get_scene_bboxes(camera_centers, output_path):
    scene_config = SceneConfig(
        origin=torch.tensor([0.0, -56.0]),
        partition_size=[60.0, 60.0],
    )
    scene = PartitionableScene(scene_config, camera_centers[..., :2])
    scene.get_bounding_box_by_camera_centers()

    scene.get_scene_bounding_box()
    save_plot(scene.plot_scene_bounding_box, osp.join(output_path, "figs", "bounding_boxes.png"))

    scene.build_partition_coordinates()
    save_plot(scene.plot_partitions, osp.join(output_path, "figs", "partitions.png"))

    return scene


def plot_pcd(xyz, colors, camera_centers, output_path, sparsify_points=16):
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    scene_size = torch.max(camera_centers, dim=0).values - torch.min(camera_centers, dim=0).values
    ax.set_xlim(
        [
            torch.min(camera_centers[:, 0]) - 0.1 * scene_size[0],
            torch.max(camera_centers[:, 0]) + 0.1 * scene_size[0],
        ]
    )
    ax.set_ylim(
        [
            torch.min(camera_centers[:, 1]) - 0.1 * scene_size[1],
            torch.max(camera_centers[:, 1]) + 0.1 * scene_size[1],
        ]
    )
    ax.scatter(
        xyz[::sparsify_points, 0],
        xyz[::sparsify_points, 1],
        c=colors[::sparsify_points] / 255.0,
        s=0.01,
    )
    ax.scatter(camera_centers[:, 0], camera_centers[:, 1], s=0.2, c="red")
    fig.savefig(osp.join(output_path, "figs", "pointcloud.png"))
    plt.close(fig)


def save_plot(func: Callable, path: str, *args, **kwargs):
    fig, ax = plt.subplots()
    func(ax, *args, **kwargs)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def main():
    args = make_parser().parse_args()
    dataset_path, manhattan_trans = args.dataset_path, args.manhattan_trans
    dataset_path = "datasets/MegaNeRF/rubble/colmap"
    manhattan_trans = "manhattan.txt"
    config_dict = vars(args)
    config_dict.pop("dataset_path")
    config_dict.pop("manhattan_trans")

    config_dict["origin"] = [float(s) for s in config_dict["origin"].split(",")]
    partition_size = config_dict["partition_size"].split(",")
    if len(partition_size) == 1:
        partition_size = partition_size * 2
    config_dict["partition_size"] = [float(s) for s in partition_size]
    config = Namespace(**config_dict)

    partition(dataset_path=dataset_path, config=config, manhattan_trans=manhattan_trans)


if __name__ == "__main__":
    main()
