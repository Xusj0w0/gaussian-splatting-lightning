**partition**

```shell
CUDA_VISIBLE_DEVICES=7 python large_scene/tools/partition.py --project_name residence-octree-feature-3_5 --dataset_path datasets/MegaNeRF/residence --partition_dim="[3,5]" --scene_config.class_path large_scene.impls.city_gaussian.UncontractCitySceneConfig --scene_config.config exp_configs/meganerf/residence/residence_implicit_lod_feature_coarse.yaml
```

**partition trianing**

```shell
CUDA_VISIBLE_DEVICES=0 python large_scene/tools/train.py -p residence-octree-3_5 --initialize_from gaussian_model.pt --max-steps 40000 --n-processes 3 --process-id 1 --config exp_configs/meganerf/residence/residence_implicit_lod_finetune.yaml --ignore-slurm
```

CUDA_VISIBLE_DEVICES=3 python large_scene/tools/train.py -p residence-octree-2 --initialize_from gaussian_model.pt --max-steps 60000 --n-processes 2 --process-id 2 --config exp_configs/meganerf/residence/residence_partition_density_controller2.yaml --ignore-slurm