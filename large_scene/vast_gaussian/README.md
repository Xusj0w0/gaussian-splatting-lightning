lightning repo的partitioning_utils.py

## 基于可见度的相机分配

基于可见度的相机分配主要有2种：
- 基于点的分配
- 基于点凸包的分配

首先需要定义一个`point_getter`函数，输入相机id，输出点:
- 若为基于点的分配，需要返回一个`points_3d`张量
    ```python
    @classmethod
    def cameras_point_based_visibilities_calculation(
        cls,
        partition_coordinates: PartitionCoordinates,
        n_cameras,
        point_getter: Callable[[int], torch.Tensor],
        device,
    ):
        partition_bounding_boxes = partition_coordinates.get_bounding_boxes(enlarge=0.).to(device=device)
        all_visibilities = torch.ones((n_cameras, len(partition_coordinates)), device="cpu") * -255.  # [N_cameras, N_partitions]

        def calculate_visibilities(camera_idx: int):
            points_3d = point_getter(camera_idx)
            visibilities, _ = Partitioning.calculate_point_based_visibilities(
                partition_bounding_boxes=partition_bounding_boxes,
                points=points_3d[..., :2],
            )  # [N_partitions]
            all_visibilities[camera_idx].copy_(visibilities.to(device=all_visibilities.device))

        from concurrent.futures.thread import ThreadPoolExecutor
        with ThreadPoolExecutor() as tpe:
            for _ in tqdm(
                    tpe.map(calculate_visibilities, range(n_cameras)),
                    total=n_cameras,
            ):
                pass

        assert torch.all(all_visibilities >= 0.)

        return all_visibilities.T  # [N_partitions, N_cameras]
    ```
- 若为基于点凸包的分配，需要返回`points_2d`, `points_3d`, `projected_points`三个张量，其中`points_2d`与`points_3d`为具有对应关系的图像点与空间点，`projected_points`为二维上的边界
    ```python
    @classmethod
    def cameras_convex_hull_based_visibilities_calculation(
        cls,
        partition_coordinates: PartitionCoordinates,
        n_cameras,
        point_getter: Callable[[int], Tuple[torch.Tensor, torch.Tensor, int]],
        enlarge: float,
        device,
    ):
        partition_bounding_boxes = partition_coordinates.get_bounding_boxes(enlarge=enlarge).to(device=device)
        all_visibilities = torch.ones((n_cameras, len(partition_coordinates)), device="cpu") * -255.  # [N_cameras, N_partitions]

        def calculate_visibilities(camera_idx: int):
            points_2d, points_3d, projected_points = point_getter(camera_idx)
            visibilities, _, _ = Partitioning.calculate_convex_hull_based_visibilities(
                partition_bounding_boxes=partition_bounding_boxes,
                points_2d=points_2d,
                points_3d=points_3d[..., :2],
                projected_points=projected_points,
            )  # [N_partitions]
            all_visibilities[camera_idx].copy_(visibilities.to(device=all_visibilities.device))

        from concurrent.futures.thread import ThreadPoolExecutor
        with ThreadPoolExecutor() as tpe:
            for _ in tqdm(
                    tpe.map(calculate_visibilities, range(n_cameras)),
                    total=n_cameras,
            ):
                pass

        assert torch.all(all_visibilities >= 0.)

        return all_visibilities.T  # [N_partitions, N_cameras]
    ```
## 基于点凸包的可见度计算

基于点凸包的计算方法：
- 计算`projected_points`的凸包，将其面积作为分母
- 计算落在`partition_bounding_boxes`中的`points_3d`的id
- 取出`points_2d`中的对应点，计算凸包，将其面积作为分子

设想：
1. points3d为相机能看到的所有3D点，points2d为3D点在相机中的图像点，bbox为一个partition的3D边界（上下为无穷），projected_points为图像4顶点
2. points3d为相机能看到的所有3D点的xy坐标，points2d为3D点在相机中的图像点，bbox为一个partition的2D边界，projected_points为图像4顶点
3. （代入VastGS）points3d为partition coordinates定义的立方体顶点（经z>0筛选），points2d为对应的投影，bbox为无穷，projected_points为图像4顶点

```python
@classmethod
def calculate_convex_hull_based_visibilities(
    cls,
    partition_bounding_boxes: MinMaxBoundingBoxes,
    points_2d: torch.Tensor,  # [N_points, 2]
    points_3d: torch.Tensor,  # [N_points, 2 or 3]
    projected_points: torch.Tensor,  # [N_points, 2]
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    assert projected_points.shape[-1] == 2

    visibilities = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.float)
    scene_convex_hull = None
    partition_convex_hull_list = []
    is_in_bounding_boxes = None
    if points_3d.shape[0] > 2:
        try:
            scene_convex_hull = ConvexHull(projected_points)
        except:
            return visibilities, scene_convex_hull, partition_convex_hull_list
        scene_area = scene_convex_hull.volume

        is_in_bounding_boxes = cls.is_in_bounding_boxes(
            bounding_boxes=partition_bounding_boxes,
            coordinates=points_3d,
        )  # [N_partitions, N_points]

        # TODO: batchify
        for partition_idx in range(is_in_bounding_boxes.shape[0]):
            if is_in_bounding_boxes[partition_idx].sum() < 3:
                partition_convex_hull_list.append(None)
                continue
            points_2d_in_partition = points_2d[is_in_bounding_boxes[partition_idx]]
            try:
                partition_convex_hull = ConvexHull(points_2d_in_partition)
            except Exception as e:
                partition_convex_hull_list.append(None)
                continue
            partition_convex_hull_list.append(partition_convex_hull)
            partition_area = partition_convex_hull.volume

            visibilities[partition_idx] = partition_area / scene_area

    return visibilities, scene_convex_hull, (partition_convex_hull_list, is_in_bounding_boxes)
```

## 基于可见度的相机分配

```python
have_assigned_cameras = torch.sum(assigned_mask, dim=-1, keepdim=True) > 0
no_assigned_cameras = ~have_assigned_cameras
have_assigned_cameras = have_assigned_cameras.to(dtype=visibilities.dtype)
no_assigned_cameras = no_assigned_cameras.to(dtype=visibilities.dtype)

max_distance_adjustments = have_assigned_cameras + no_camera_enlarge_distance * no_assigned_cameras
visibility_adjustments = have_assigned_cameras + ((1. / no_camera_reduce_threshold) * no_assigned_cameras)
```
对于按位置分配不成功的相机给予一定优待：设置更大的距离阈值与更小的可见度阈值
但是这里按照可见度的分配和VastGS的逻辑有点不一样，这里的可见度分配结果是去除了相机位置分配的结果的，若相机都已经分配了，就不会再按可见度分配给别的partition了
需要修改