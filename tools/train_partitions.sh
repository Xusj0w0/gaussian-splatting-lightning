# use tmux to manage multiple `train_partitions.py` tasks

PWD=$(pwd)

export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/utils${PYTHONPATH:+:${PYTHONPATH}}"

ENV_VARS=(
    PATH
    LD_LIBRARY_PATH
    PYTHONPATH
)

CUDA_LIST=(2 3 5)
NUM_PROC=${#CUDA_LIST[@]}

PARTITION_DATA_PATH="$PWD/datasets/chongqing/colmap/partitions-size_2.0-enlarge_0.1-point_visibility_0.9_0.11"
PROJECT_NAME="chongqing"

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
        python $PWD/utils/train_colmap_partitions_v2.py \
            $PARTITION_DATA_PATH \
            -p $PROJECT_NAME \
            --config configs/gsplat_v1.yaml \
            --n-processes $NUM_PROC \
            --process-id $((i + 1)) \
            -- \
            --logger tensorboard \
        " C-m
done
