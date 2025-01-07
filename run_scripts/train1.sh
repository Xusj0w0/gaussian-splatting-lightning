export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

PARTITION_DATA_PATH="/data/xusj/Projects/3drec/gspl/datasets/rubble/colmap/partitions-size_60.0-enlarge_0.1-visibility_0.9_0.25"
PROJECT_NAME="MegaNeRF-rubble"
export CUDA_VISIBLE_DEVICES="0"
cd /data/xusj/Projects/3drec/gspl/repos/gaussian-splatting-lightning
python utils/train_colmap_partitions_v2.py \
    ${PARTITION_DATA_PATH} \
    -p ${PROJECT_NAME} \
    --scalable-config utils/scalable_param_configs/appearance.yaml \
    --config configs/appearance_embedding_renderer/sh_view_dependent-mip.yaml \
    -- \
    --data.parser.appearance_groups appearance_image_dedicated \
    --model.gaussian.optimization.spatial_lr_scale 15 \
    --data.parser.down_sample_factor 3 \
    --logger tensorboard