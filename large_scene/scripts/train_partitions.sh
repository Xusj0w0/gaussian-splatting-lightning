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

CUDA_LIST=(1 2 3)
NUM_PROC=${#CUDA_LIST[@]}

DATASET_PATH=$PWD/datasets/MatrixCity/small_city/aerial/colmap
PARTITION_DATA_PATH="$PWD/outputs/residence/partition_infos"
PROJECT_NAME="residence"
TRAINING_SCRIPT="$PWD/large_scene/CityGaussian/partition_training.py"

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
        python $TRAINING_SCRIPT \
            $PARTITION_DATA_PATH \
            -p $PROJECT_NAME \
            --dataset_path $DATASET_PATH \
            --eval \
            --n-processes $NUM_PROC \
            --process-id $((i + 1)) \
            -- \
            --data.parser.down_sample_factor 4 \
            --logger tensorboard \
        " C-m
done
