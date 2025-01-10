export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/data/xusj/Projects/3drec/gaussian-splatting-lightning:/data/xusj/Projects/3drec/gaussian-splatting-lightning/large_scene/VastGaussian"

CUDA_VISIBLE_DEVICES="1" python main.py fit \
    --project "compare_appearance" \
    -n "wo-apperance" \
    --config "configs/gsplat_v1.yaml" \
    --data.path "datasets/Mip-NeRF360/v2/bicycle" \
    --data.parser "Colmap" \
    --data.parser.down_sample_factor "2"

# CUDA_VISIBLE_DEVICES="2" python main.py fit \
#     --project "compare_appearance" \
#     -n "gspl-appearance" \
#     --config "configs/appearance_embedding_renderer/sh_view_dependent.yaml" \
#     --data.path "datasets/Mip-NeRF360/v2/bicycle" \
#     --data.parser "Colmap" \
#     --data.parser.down_sample_factor "2" \
#     --data.parser.appearance_groups "dedicated"

CUDA_VISIBLE_DEVICES="3" python main.py fit \
    --project "compare_appearance" \
    -n "vastgs-appearance" \
    --config "large_scene/VastGaussian/configs/gsplat_appearance_modeling.yaml" \
    --data.path "datasets/Mip-NeRF360/v2/bicycle" \
    --data.parser "Colmap" \
    --data.parser.down_sample_factor "2" \
    --data.parser.appearance_groups "dedicated"