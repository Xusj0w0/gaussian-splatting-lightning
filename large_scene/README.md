**partition**

```shell
CUDA_VISIBLE_DEVICES=7 python large_scene/tools/partition.py --project_name residence-feat-part6 --dataset_path datasets/MegaNeRF/residence --partition_dim="[2,4]" --scene_config.class_path large_scene.impls.grid_gaussian.GridSceneConfig --scene_config.train_config exp_configs/meganerf/residence/coarse_feature.yaml
```

**partition training**

```shell
CUDA_VISIBLE_DEVICES=7 python large_scene/tools/train.py --ignore-slurm -p part-bs2-3-2 --initialize_from gaussian_model.pt --max-steps 60000 --n-processes 2 --process-id 2 --config exp_configs/meganerf/residence/finetune.yaml
```
CUDA_VISIBLE_DEVICES=4 python large_scene/tools/train.py --ignore-slurm -p hashgrid-part --initialize_from gaussian_model.pt --max-steps 60000 --n-processes 2 --process-id 2 --config exp_configs/meganerf/residence/hashgrid_finetune.yaml