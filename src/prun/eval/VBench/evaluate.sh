#!/bin/bash

# Define the model list
#models=("inference_epoch_001" "inference_epoch_002" "inference_epoch_003" "inference_epoch_004" "inference_epoch_005")
#models=("epoch_001" "epoch_002" "epoch_003" "epoch_004" "epoch_005" "epoch_006" "epoch_007" "epoch_008" "epoch_009" "epoch_010" "epoch_011" "epoch_012")
port_number=29513
export MASTER_PORT=$port_number

#models=("inference_epoch_001" "inference_epoch_002" "inference_epoch_003" "inference_epoch_004" "inference_epoch_005")
# Define the dimension list
#dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
dimensions=("subject_consistency")
# Corresponding folder names
folders=("subject_consistency" "scene" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")

# Base path for videos
base='/home/dreamyou070/Prun/src/prun'
#base_folder="${base}/experiment/EX_0_teacher/teacher"

# Loop over each model
#for model in "${models[@]}"; do
    # Loop over each dimension
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}
    folder=${folders[i]}
    # Construct the video path
    videos_path="/scratch2/dreamyou070/Prun/result/1_teacher_calibration_data/sample"
    output_path="/scratch2/dreamyou070/Prun/result/1_teacher_calibration_data/sample_result"
    echo "$dimension $videos_path"
    # Run the evaluation script
    python ${base}/eval/VBench/evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path --mode=custom_input --save_name "${model}_${dimension}"
done
#done
