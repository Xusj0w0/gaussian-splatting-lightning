**partition**

```shell
CUDA_VISIBLE_DEVICES=7 python large_scene/tools/partition.py --project_name residence-feat-part6 --dataset_path datasets/MegaNeRF/residence --partition_dim="[2,4]" --scene_config.class_path large_scene.impls.grid_gaussian.GridSceneConfig --scene_config.train_config exp_configs/meganerf/residence/coarse_feature.yaml
```

**partition trianing**

```shell
CUDA_VISIBLE_DEVICES=5 python large_scene/tools/train.py --ignore-slurm -p residence-feat-part4 --initialize_from gaussian_model.pt --max-steps 60000 --n-processes 2 --process-id 2 --config exp_configs/meganerf/residence/finetune_feature.yaml
```

CUDA_VISIBLE_DEVICES=1 python large_scene/tools/train.py --ignore-slurm -p residence-octree-feature --initialize_from gaussian_model.pt --max-steps 60000 --n-processes 2 --process-id 2 --config exp_configs/meganerf/residence/residence_feature_finetune.yaml
CUDA_VISIBLE_DEVICES=1 python large_scene/tools/partition.py --project_name residence-octree-feature2 --dataset_path datasets/MegaNeRF/residence --partition_dim="[2,4]" --scene_config.class_path large_scene.impls.city_gaussian.UncontractAllVisCitySceneConfig --scene_config.config exp_configs/meganerf/residence/residence_feature_coarse.yaml