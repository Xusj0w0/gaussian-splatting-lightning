export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES="1"
PROJECT_NAME=shanxi
EXP_NAME=shanxi-gsplat-shvd-sfmer-denser
MAX_STEPS=60000

python main.py fit \
    --viewer \
    --project $PROJECT_NAME \
    -n $EXP_NAME \
    --trainer.max_steps $MAX_STEPS \
    --model.gaussian.max_steps $MAX_STEPS \
    --model.renderer.max_steps $MAX_STEPS \
    --config configs/appearance_embedding_renderer/sh_view_dependent.yaml \
    --model.density.densification_interval 200 \
    --model.density.densify_from_iter 1000 \
    --model.density.densify_until_iter 50000 \
    --model.density.densify_grad_threshold 0.0001 \
    --data.path /data/xusj/Projects/3drec/gspl/datasets/shanxi-building/colmap/undistorted \
    --data.parser Colmap \
    --data.train_max_num_images_to_cache 512 \
    --data.async_caching true \
    --data.parser.down_sample_factor 2 \
    --logger tensorboard