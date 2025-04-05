**partition trianing**

```shell
CUDA_VISIBLE_DEVICES=1 python large_scene/tools/train.py -p residence-octree-3_5 --max-steps 60000 --n-processes 3 --process-id 1 --config exp_configs/meganerf/residence_implicit_lod_finetune.yaml --ignore-slurm
```