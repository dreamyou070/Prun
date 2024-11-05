#!/bin/bash

# Define the model list
#models=("inference_epoch_001" "inference_epoch_002" "inference_epoch_003" "inference_epoch_004" "inference_epoch_005")
models=("epoch_012" "epoch_013" "epoch_014")
port_number=20002
export MASTER_PORT=$port_number

#models=("inference_epoch_001" "inference_epoch_002" "inference_epoch_003" "inference_epoch_004" "inference_epoch_005")
# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "motion_smoothness" "dynamic_degree" "aesthetic_quality" "imaging_quality")
# folders=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
# Corresponding folder names


# Base path for videos
base='/share0/dreamyou070/dreamyou070/LIFOVideo2'
base_folder="${base}/experiment/EX_5_panda_only_high_raft_data/experiment_remove_layer_6/samples"

# Loop over each model
for model_idx in "${models[@]}"; do
    # Loop over each dimension
    model=${models[model_idx]}
    for i in "${!dimensions[@]}"; do
        # Get the dimension and corresponding folder
        dimension=${dimensions[i]}
        # Construct the video path
        videos_path="${base_folder}/${model}"
        output_path="${base_folder}"
        echo "$dimension $videos_path"
        # Run the evaluation script
        python ${base}/eval/VBench/evaluate.py \
          --videos_path $videos_path \
          --dimension $dimension \
          --output_path $base_folder \
          --mode=custom_input \
          --save_name "${model}_${dimension}"
    done
done