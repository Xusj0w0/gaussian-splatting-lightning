export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
CUR_DIR=$(pwd)
NUM_PROC=4
CUDA_LIST=(1 2 3 4)

kill_pids() {
  for pid in "$@"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        echo "Killed process $pid"
    fi
  done
}

PARTITION_DATA_PATH="/data/xusj/Projects/3drec/gspl/datasets/rubble/colmap/partitions-size_60.0-enlarge_0.1-visibility_0.9_0.25"

PROJECT_NAME="MegaNeRF-rubble-mip"
mkdir -p logs/$PROJECT_NAME
PIDS=()
for i in "${!CUDA_LIST[@]}"; do
    logfile="$CUR_DIR/logs/$PROJECT_NAME/process_$i.log"

    export CUDA_VISIBLE_DEVICES="${CUDA_LIST[$i]}"
    cd /data/xusj/Projects/3drec/gspl/repos/gaussian-splatting-lightning
    python utils/train_colmap_partitions_v2.py \
        ${PARTITION_DATA_PATH} \
        -p ${PROJECT_NAME} \
        --scalable-config utils/scalable_param_configs/appearance.yaml \
        --config configs/appearance_embedding_renderer/sh_view_dependent-mip.yaml \
        --n-processes 4 \
        --process-id $((i+1)) \
        -- \
        --data.parser.appearance_groups appearance_image_dedicated \
        --model.gaussian.optimization.spatial_lr_scale 15 \
        --data.parser.down_sample_factor 3 \
        --logger tensorboard \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    echo "Started process $i with PID $!, logging to $logfile"
done

trap 'echo "Stopping all background processes..."; kill_pids "${PIDS[@]}"' EXIT
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "Process $pid has completed."
done

echo "All processes have completed."