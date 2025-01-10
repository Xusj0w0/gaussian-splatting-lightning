COLMAP_CMD=$(which mycolmap)

for DATADIR in "/data/xusj/Datasets/Dataset-3drec/Mill-19/building" "/data/xusj/Datasets/Dataset-3drec/Mill-19/rubble"; do
    python tools/meganerf2colmap.py \
        /data/xusj/Datasets/Dataset-3drec/Mill-19/building \
        --colmap_executable $COLMAP_CMD \
        --refine \
        --gpu_id "1"
done

python tools/copy_images.py \
    --image_path /data/xusj/Datasets/Dataset-3drec/UrbanScene3D/orig_images/Residence/photos \
    --dataset_path /data/xusj/Datasets/Dataset-3drec/UrbanScene3D/processed/residence

python tools/copy_images.py \
    --image_path /data/xusj/Datasets/Dataset-3drec/UrbanScene3D/orig_images/Sci-Art/photos \
    --dataset_path /data/xusj/Datasets/Dataset-3drec/UrbanScene3D/processed/sci-art

for DATADIR in residence sci-art campus; do
    python tools/meganerf2colmap.py \
        /data/xusj/Datasets/Dataset-3drec/UrbanScene3D/processed/$DATADIR \
        --colmap_executable $COLMAP_CMD \
        --refine \
        --gpu_id "1"
done
