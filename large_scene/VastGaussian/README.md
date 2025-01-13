- `partition.py`：区域分块
- `train_partitions.py`：分块训练
- `train_partitions.sh`：使用tmux管理分块训练任务
- `merge.py`：还未完成

原始高斯实现：若设置了eval为True，从colmap结果中加载图像，按名称排序后间隔llffhold步取一张用来测试，训练与测试不重合
gspl实现：设置eval_image_select_model为"step"，并设置split_mode为"experiment"（不重复）可以达到上述同样的效果
但是到分块重建会有问题：
1. VastGaussian的实现：在所有图像上按照llffhold分割后，对train_cameras进行区域分配
2. 相应的，gspl应该如此实现：
    1. 先将所有camera加进来进行划分（修改了meganerf2colmap.py，可以在转换的时候把训练集和测试集列出来）
    2. 每一块单独训练时：
        1. 加载数据先指定image_list为这一块的全部相机
        2. 设置eval_image_select_mode为list
        3. 指定eval_list为val_cameras.txt的路径
        4. 设置split_mode为"experiment"（不向train_list中添加val_cameras中的相机）
        5. 注：这样会导致colmap parser解析时len(eval_image_set) != 0，但由于split_mode不为"list"，因此不会报错，仅会警告
    3. 合并
    4. 最终测试：不设置image_list，但是需要设置eval_image_select_mode为list，指定eval_list为val_cameras.txt的路径，设置split_mode为"experiment"
    
