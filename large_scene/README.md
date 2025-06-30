**partition**

```shell
CUDA_VISIBLE_DEVICES=7 python large_scene/tools/partition.py --project_name residence-partition --dataset_path datasets/MegaNeRF/residence --partition_dim="[2,4]" --scene_config.class_path large_scene.impls.grid_gaussian.HashGridSceneConfig --scene_config.visibility_threshold 0.3
```

**partition training**

```shell
CUDA_VISIBLE_DEVICES=4 python large_scene/tools/train.py --ignore-slurm -p residence-partition --initialize_from gaussian_model.pt --max-steps 60_000 --n-processes 2 --process-id 2 --config exp_configs/meganerf/residence/hashgrid_finetune.yaml
```

```shell
python large_scene/tools/merge_hash.py -p residence-partition
```

```shell
CUDA_VISIBLE_DEVICES=1 python large_scene/tools/evaluate.py --ckpt outputs/residence-partition/merged/merged.ckpt --output outputs/residence-partition/merged/evaluations
```