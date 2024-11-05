#!/bin/bash
# Define the dimension list
export MASTER_PORT=29517
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "motion_smoothness" "dynamic_degree")
folders=("inference_epoch_001" "inference_epoch_002" "inference_epoch_003" "inference_epoch_004" "inference_epoch_005")
# Base path for videos
base='/share0/dreamyou070/dreamyou070/LIFOVideo2'
video_path_base=${base}/experiment/EX_4_first_method/experiment_layer_num_7/inference
base_dir=${base}/eval/VBench

# Loop over each model
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}
    for ii in "${!folders[@]}"; do
        folder=${folders[ii]}
        videos_path="${video_path_base}/${folder}/samples_vbench"
        output_path="${video_path_base}"
        echo "$dimension to ${output_path}"
        save_name="${folder}_${dimension}"
        python ${base_dir}/evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path --mode=custom_input --save_name $save_name
    done
done