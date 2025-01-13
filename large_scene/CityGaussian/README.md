参考CityGS中script/run_citygs.sh，流程大致如下：
1. 粗训练整个场景
2. 数据分块
3. 精细微调

在coarse的设置中：
- building和mc_aerial的position_lr和scale_lr与其他的有所区别，其他的都一样
- 只有mc的分辨率为原尺寸，其他的降采样4倍
- gspl-lightning的默认参数设置与citygs的一致
- 位置lr_scheduler也一致，但是city_gs没有warmup

coarse实现时将image_list设置为train_cameras.txt+reconstruction（能否不设置image_list，将eval_...设置为val_cameras.txt+experiment？）