import copy
import enum
import math
import os
import os.path as osp
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scene.cameras import SimpleCamera
from scene.dataset_readers import storePly
from scene.vastgs.data_partition import (CameraPartition, CameraPose,
                                         ProgressiveDataPartitioning)
from scene.vastgs.graham_scan import run_graham_scan

from internal.utils.partitioning_utils import (MinMaxBoundingBoxes,
                                               PartitionCoordinates)
from utils.graphics_utils import BasicPointCloud


@dataclass
class VastGSPartitionCoordinates(PartitionCoordinates):
    # xy: torch.Tensor, (N_partitions, 4), (xmin, zmin, xmax, zmax)
    def get_bounding_boxes(self, size: Tuple[float], enlarge: float = 0.0) -> MinMaxBoundingBoxes:
        return MinMaxBoundingBoxes(min=self.xy[:, :2], max=self.xy[:, 2:])

    def get_str_id(self, idx):
        id_tensor = self.id[idx]
        x, y = id_tensor[0], id_tensor[1]
        if isinstance(id, torch.Tensor):
            x, y = x.item(), y.item()
        return "{:d}_{:d}".format(x, y)


class VastGSProgressiveDataPartitioning(ProgressiveDataPartitioning):
    """
    Modified from `ProgressiveDataPartitioning`
    """

    def __init__(
        self,
        pcd,
        train_cameras,
        model_path,
        m_region=2,
        n_region=4,
        extend_rate=0.2,
        visible_rate=0.25,
        extra_trans: Optional[np.ndarray] = None,
    ):
        self.partition_scene = None
        self.pcd = pcd
        # print(f"self.pcd={self.pcd}")
        self.model_path = model_path  # 存放模型位置
        self.partition_dir = os.path.join(model_path, "partition_point_cloud")
        # self.partition_ori_dir = os.path.join(self.partition_dir, "ori")
        # self.partition_extend_dir = os.path.join(self.partition_dir, "extend")
        self.partition_visible_dir = os.path.join(self.partition_dir, "visible")
        self.save_partition_data_dir = os.path.join(self.model_path, "partition_data.pkl")
        os.makedirs(self.partition_visible_dir, exist_ok=True)

        self.m_region = m_region
        self.n_region = n_region
        self.extend_rate = extend_rate
        self.visible_rate = visible_rate
        self.extra_trans = extra_trans
        self.fig, self.ax = self.draw_pcd(self.pcd, train_cameras)
        self.run_DataPartition(train_cameras)

    def run_DataPartition(self, train_cameras):
        if not os.path.exists(self.save_partition_data_dir):
            partition_dict = self.Camera_position_based_region_division(train_cameras)
            partition_dict, refined_ori_bbox = self.refine_ori_bbox(partition_dict)
            # partition_dict, refined_ori_bbox = self.refine_ori_bbox_average(partition_dict)
            partition_list = self.Position_based_data_selection(partition_dict, refined_ori_bbox)

            # location_based_assignments
            self.location_based_assignments = self.get_assignments(partition_list, train_cameras)

            self.draw_partition(partition_list)
            self.partition_scene = self.Visibility_based_camera_selection(partition_list)

            # final_assignments
            self.final_assignments = self.get_assignments(partition_list, train_cameras)

            self.save_partition_data()
        else:
            self.partition_scene = self.load_partition_data()

    def get_assignments(self, partition_list: List[CameraPartition], all_cameras: List[SimpleCamera]):
        assignments = np.zeros((self.m_region * self.n_region, len(all_cameras)), dtype=bool)
        for pid, partition in enumerate(partition_list):
            images_in_partition = [camera.camera.image_name for camera in partition.cameras]
            for cid, camera in enumerate(all_cameras):
                if camera.image_name in images_in_partition:
                    assignments[pid, cid] = True
        return assignments

    def draw_pcd(self, pcd, train_cameras):
        x_coords = pcd.points[::16, 0]
        z_coords = pcd.points[::16, 2]
        colors = pcd.colors[::16]
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(x_coords, z_coords, c=(colors), s=1)
        ax.title.set_text("Plot of 2D Points")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Z-axis")
        fig.tight_layout()
        fig.savefig(os.path.join(self.model_path, "pcd.png"), dpi=200)
        x_coords = np.array([cam.camera_center[0].item() for cam in train_cameras])
        z_coords = np.array([cam.camera_center[2].item() for cam in train_cameras])
        ax.scatter(x_coords, z_coords, color="red", s=1)
        fig.savefig(os.path.join(self.model_path, "camera_on_pcd.png"), dpi=200)
        return fig, ax

    def save_partition_data(self):
        partition_dict = {
            "partition_scene": self.partition_scene,
            "location_base_assignments": self.location_based_assignments,
            "final_assignments": self.final_assignments,
        }
        with open(self.save_partition_data_dir, "wb") as f:
            pickle.dump(partition_dict, f)

    def load_partition_data(self):
        with open(self.save_partition_data_dir, "rb") as f:
            partition_dict = pickle.load(f)
        self.location_based_assignments = partition_dict["location_base_assignments"]
        self.final_assignments = partition_dict["final_assignments"]
        return partition_dict["partition_scene"]

    def Position_based_data_selection(
        self, partition_dict: Dict, refined_ori_bbox: Dict  # , all_cameras: List[SimpleCamera]
    ):
        """
        2.基于位置的数据选择
        思路: 1.计算每个partition的x z边界
             2.然后按照extend_rate将每个partition的边界坐标扩展, 得到新的边界坐标 [x_min, x_max, z_min, z_max]
             3.根据extend后的边界坐标, 获取该部分对应的点云
        问题: 有可能根据相机确定边界框后, 仍存在一些比较好的点云没有被选中的情况, 因此extend_rate是一个超参数, 需要根据实际情况调整
        :return partition_list: 每个部分对应的点云，所有相机，边界
        """
        # 计算每个部分的拓展后的边界坐标，以及该部分对应的点云
        pcd = self.pcd
        partition_list = []
        point_num = 0
        point_extend_num = 0
        # self.location_based_assignments = np.zeros(
        #     (self.m_region * self.n_region, len(all_cameras)), dtype=np.float32
        # )12

        for pid, (partition_idx, camera_list) in enumerate(partition_dict.items()):
            min_x, max_x, min_z, max_z = refined_ori_bbox[partition_idx]
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            extend_camera_bbox = [
                min_x - self.extend_rate * (max_x - min_x),
                max_x + self.extend_rate * (max_x - min_x),
                min_z - self.extend_rate * (max_z - min_z),
                max_z + self.extend_rate * (max_z - min_z),
            ]
            # print(
            #     "Partition",
            #     partition_idx,
            #     "ori_camera_bbox",
            #     ori_camera_bbox,
            #     "\textend_camera_bbox",
            #     extend_camera_bbox,
            # )
            ori_camera_centers = []
            for camera_pose in camera_list:
                ori_camera_centers.append(camera_pose.pose)

            # 保存ori相机位置
            # storePly(
            #     os.path.join(self.partition_ori_dir, f"{partition_idx}_camera_centers.ply"),
            #     np.array(ori_camera_centers),
            #     np.zeros_like(np.array(ori_camera_centers)),
            # )

            # TODO: 需要根据拓展后的边界重新添加相机
            # modified:
            new_camera_list = []
            extend_camera_centers = []
            for id, camera_list in partition_dict.items():
                for camera_pose in camera_list:
                    if (
                        extend_camera_bbox[0] <= camera_pose.pose[0] <= extend_camera_bbox[1]
                        and extend_camera_bbox[2] <= camera_pose.pose[2] <= extend_camera_bbox[3]
                    ):
                        extend_camera_centers.append(camera_pose.pose)
                        new_camera_list.append(camera_pose)
            # for cid, camera in enumerate(all_cameras):
            #     pose = camera.camera_center.cpu()
            #     if (
            #         extend_camera_bbox[0] <= pose[0] <= extend_camera_bbox[1]
            #         and extend_camera_bbox[2] <= pose[2] <= extend_camera_bbox[3]
            #     ):
            #         extend_camera_centers.append(pose)
            #         new_camera_list.append(CameraPose(camera=camera, pose=pose))
            #         self.location_based_assignments[pid, cid] = True

            # 保存extend后新添加的相机位置
            # storePly(
            #     os.path.join(self.partition_extend_dir, f"{partition_idx}_camera_centers.ply"),
            #     np.array(extend_camera_centers),
            #     np.zeros_like(np.array(extend_camera_centers)),
            # )

            # 获取该部分对应的点云
            points, colors, normals = self.extract_point_cloud(
                pcd, ori_camera_bbox
            )  # 分别提取原始边界内的点云，和拓展边界后的点云
            points_extend, colors_extend, normals_extend = self.extract_point_cloud(pcd, extend_camera_bbox)
            # 论文中说点云围成的边界框的高度选取为最高点到地平面的距离，但在本实现中，因为不确定地平面位置，(可视化中第平面不用坐标轴xz重合)
            # 因此使用整个点云围成的框作为空域感知的边界框
            partition_list.append(
                CameraPartition(
                    partition_id=partition_idx,
                    cameras=new_camera_list,
                    point_cloud=BasicPointCloud(points_extend, colors_extend, normals_extend),
                    ori_camera_bbox=ori_camera_bbox,
                    extend_camera_bbox=extend_camera_bbox,
                    extend_rate=self.extend_rate,
                    ori_point_bbox=self.get_point_range(points),
                    extend_point_bbox=self.get_point_range(points_extend),
                )
            )

            point_num += points.shape[0]
            point_extend_num += points_extend.shape[0]
            # storePly(
            #     os.path.join(self.partition_ori_dir, f"{partition_idx}.ply"), points, colors
            # )  # 分别保存未拓展前 和 拓展后的点云
            # storePly(
            #     os.path.join(self.partition_extend_dir, f"{partition_idx}_extend.ply"), points_extend, colors_extend
            # )

        # 未拓展边界前：根据位置选择后的数据量会比初始的点云数量小很多，因为相机围成的边界会比实际的边界小一些，因此使用这些边界筛点云，点的数量会减少
        # 拓展边界后：因为会有许多重合的点，因此点的数量会增多
        # print(
        #     f"Total ori point number: {pcd.points.shape[0]}\n",
        #     f"Total before extend point number: {point_num}\n",
        #     f"Total extend point number: {point_extend_num}\n",
        # )

        return partition_list

    def Visibility_based_camera_selection(self, partition_list):
        """3.基于可见性的相机选择 和 基于覆盖率的点选择
        思路：引入空域感知的能见度计算
            1.假设当前部分为i，选择j部分中的相机，
            2.将i部分边界框投影到j中的相机中，得到投影区域的面积（边界框只取地上的部分，并且可以分成拓展前和拓展后两种边界框讨论）
            3.计算投影区域面积与图像像素面积的比值，作为能见度
            4.将j中能见度大于阈值的相机s加入i中
            5.将j中所有可以投影到相机s的点云加入到i中
        :param visible_rate: 能见度阈值 默认为0.25 同论文
        """
        # 复制一份新的变量，用于添加可视相机后的每个部分的所有相机
        # 防止相机和点云被重复添加
        add_visible_camera_partition_list = copy.deepcopy(partition_list)
        client = 0
        for idx, partition_i in enumerate(partition_list):  # 第i个partition
            new_points = []  # 提前创建空的数组 用于保存新增的点
            new_colors = []
            new_normals = []

            pcd_i = partition_i.point_cloud
            partition_id_i = partition_i.partition_id  # 获取当前partition的编号
            # 获取当前partition中点云围成的边界框的8角坐标
            partition_ori_point_bbox = partition_i.ori_point_bbox
            partition_extend_point_bbox = partition_i.extend_point_bbox
            ori_8_corner_points = self.get_8_corner_points(
                partition_ori_point_bbox
            )  # 获取点云围成的边界的8个角点的坐标
            extent_8_corner_points = self.get_8_corner_points(partition_extend_point_bbox)

            corner_points = []
            for point in extent_8_corner_points.values():
                corner_points.append(point)
            # storePly(
            #     os.path.join(self.partition_extend_dir, f"{partition_id_i}_corner_points.ply"),
            #     np.array(corner_points),
            #     np.zeros_like(np.array(corner_points)),
            # )

            total_partition_camera_count = 0  # 当前partition中的相机数量
            for partition_j in partition_list:  # 第j个partiiton
                partition_id_j = partition_j.partition_id  # 获取当前partition的编号
                if partition_id_i == partition_id_j:
                    continue  # 如果当前partition与之前相同，则跳过
                print(f"Now processing partition i:{partition_id_i} and j:{partition_id_j}")
                # 获取当前partition中的点云
                pcd_j = partition_j.point_cloud

                append_camera_count = 0  # 用于记录第j个parition被添加了个新相机
                # 依次获取第j个partition中每个相机的投影矩阵
                # Visibility_based_camera_selection
                for cameras_pose in partition_j.cameras:
                    camera = cameras_pose.camera  # 获取当前相机
                    # 将i区域的点云投影到相机平面
                    # 3D points distributed on the object surface
                    # _, points_in_image, _ = self.point_in_image(camera, pcd_i.points)
                    # if not len(points_in_image) > 3: continue

                    # 将i部分的point_cloud边界框投影到j的当前相机中
                    # Visibility_based_camera_selection
                    # airspace-aware visibility
                    proj_8_corner_points = {}
                    for key, point in extent_8_corner_points.items():
                        points_in_image, _, _ = self.point_in_image(camera, np.array([point]))
                        if len(points_in_image) == 0:
                            continue
                        proj_8_corner_points[key] = points_in_image[0]

                    # 基于覆盖率的点选择
                    # i部分中点云边界框投影在j部分当前图像中的面积与当前图像面积的比值
                    if not len(list(proj_8_corner_points.values())) > 3:
                        continue
                    pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)
                    # pkg = run_graham_scan(points_in_image, camera.image_width, camera.image_height)
                    if pkg["intersection_rate"] >= self.visible_rate:
                        collect_names = [
                            camera_pose.camera.image_name
                            for camera_pose in add_visible_camera_partition_list[idx].cameras
                        ]
                        if cameras_pose.camera.image_name in collect_names:
                            # print("skip")
                            continue  # 如果相机已经存在，则不需要再重复添加
                        append_camera_count += 1
                        # print(f"Partition {idx} Append Camera {camera.image_name}")
                        # 如果空域感知比率大于阈值，则将j中的当前相机添加到i部分中
                        add_visible_camera_partition_list[idx].cameras.append(cameras_pose)
                        # 筛选在j部分中的所有点中哪些可以投影在当前图像中
                        # Coverage-based point selection
                        _, _, mask = self.point_in_image(camera, pcd_j.points)  # 在原始点云上需要新增的点
                        updated_points, updated_colors, updated_normals = (
                            pcd_j.points[mask],
                            pcd_j.colors[mask],
                            pcd_j.normals[mask],
                        )
                        # 更新i部分的需要新增的点云，因为有许多相机可能会观察到相同的点云，因此需要对点云进行去重
                        new_points.append(updated_points)
                        new_colors.append(updated_colors)
                        new_normals.append(updated_normals)

                        with open(os.path.join(self.model_path, "graham_scan"), "a") as f:
                            f.write(
                                f"intersection_area:{pkg['intersection_area']}\t"
                                f"image_area:{pkg['image_area']}\t"
                                f"intersection_rate:{pkg['intersection_rate']}\t"
                                f"partition_i:{partition_id_i}\t"
                                f"partition_j:{partition_id_j}\t"
                                f"append_camera_id:{camera.image_name}\t"
                                f"append_camera_count:{append_camera_count}\n"
                            )
                total_partition_camera_count += append_camera_count

            with open(os.path.join(self.model_path, "partition_cameras"), "a") as f:
                f.write(
                    f"partition_id:{partition_id_i}\t"
                    f"total_append_camera_count:{total_partition_camera_count}\t"
                    f"total_camera:{len(add_visible_camera_partition_list[idx].cameras)}\n"
                )

            camera_centers = []
            for camera_pose in add_visible_camera_partition_list[idx].cameras:
                camera_centers.append(camera_pose.pose)

            # 保存相机坐标，用于可视化相机位置
            storePly(
                os.path.join(self.partition_visible_dir, f"{partition_id_i}_camera_centers.ply"),
                np.array(camera_centers),
                np.zeros_like(np.array(camera_centers)),
            )

            # 点云去重
            point_cloud = add_visible_camera_partition_list[idx].point_cloud
            new_points.append(point_cloud.points)
            new_colors.append(point_cloud.colors)
            new_normals.append(point_cloud.normals)
            new_points = np.concatenate(new_points, axis=0)
            new_colors = np.concatenate(new_colors, axis=0)
            new_normals = np.concatenate(new_normals, axis=0)

            new_points, mask = np.unique(new_points, return_index=True, axis=0)
            new_colors = new_colors[mask]
            new_normals = new_normals[mask]

            # 当第j部分所有相机都筛选完之后，更新最终的点云
            add_visible_camera_partition_list[idx] = add_visible_camera_partition_list[idx]._replace(
                point_cloud=BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals)
            )  # 更新点云，新增的点云有许多重复的点，需要在后面剔除掉
            storePly(
                os.path.join(self.partition_visible_dir, f"{partition_id_i}_visible.ply"),
                new_points,
                np.clip(new_colors * 255, 0, 255).astype(np.uint8),
            )  # 保存可见性选择后每个partition的点云

        return add_visible_camera_partition_list

    def save_reverted_pcds(self, init_pcd_dir: str):
        if not osp.exists(init_pcd_dir):
            os.makedirs(init_pcd_dir)
        self.partition_scene: List[CameraPartition]
        transform = np.linalg.inv(self.extra_trans)
        Rt, t = transform[:3, :3].T, transform[:3, -1]
        for idx, partition in enumerate(self.partition_scene):
            xyz, rgb = partition.point_cloud.points, partition.point_cloud.colors
            xyz_reverted = xyz @ Rt + t
            storePly(
                osp.join(init_pcd_dir, f"{partition.partition_id}.ply"),
                xyz_reverted,
                np.clip(rgb * 255, 0, 255).astype(np.uint8),
            )


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
