PWD=$(pwd)

export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/utils${PYTHONPATH:+:${PYTHONPATH}}"

ENV_VARS=(
    PATH
    LD_LIBRARY_PATH
    PYTHONPATH
)

NUM_PROC=3
CUDA_LIST=(1 2 4)

DATASET_NAME=rubble
DATASET_PREFIX=MegaNeRF
PARTITION_DATA_PATH="$PWD/tmp/partitions/$DATASET_NAME/partitions"
PROJECT_NAME="$DATASET_NAME-vastgs"

for i in "${!CUDA_LIST[@]}"; do
    # launch tmux session
    SESSION_NAME="${PROJECT_NAME}-${i}"
    tmux new-session -d -s $SESSION_NAME
    tmux set-option -t $SESSION_NAME remain-on-exit on

    # set environment variables
    for VAR in "${ENV_VARS[@]}"; do
        VAR_VAL=${!VAR}
        tmux send-keys -t $SESSION_NAME "export $VAR=$VAR_VAL" C-m
    done
    tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=${CUDA_LIST[$i]}" C-m

    # execute commands
    tmux send-keys -t $SESSION_NAME \
        " \
        python large_scene/VastGaussian/train_partitions.py \
            $PARTITION_DATA_PATH \
            -p $PROJECT_NAME \
            --dataset_path $PWD/datasets/$DATASET_PREFIX/$DATASET_NAME/colmap \
            --eval \
            --config configs/gsplat_v1.yaml \
            --n-processes $NUM_PROC \
            --process-id $((i + 1)) \
            -- \
            --model.gaussian.optimization.spatial_lr_scale 15 \
            --data.parser.down_sample_factor 2 \
            --logger tensorboard \
        " C-m
done
