#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=/home/dreamyou070/VideoDistill/logs/20240919_eval.log
#SBATCH --error=/home/dreamyou070/VideoDistill/logs/20240919_eval.log
#SBATCH --time=48:00:00

# Define the dimension list
export MASTER_PORT=29522
#dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "motion_smoothness" "dynamic_degree")
dimensions=("dynamic_degree")

# Loop over each model
for i in "${!dimensions[@]}"; do
    dimension=${dimensions[i]}
    videos_path="/scratch2/dreamyou070/Prun/result/3_animatelcm_algorithmic_searcher_prun_10_blocks_50_prompts/teacher_folder"
    output_path="/scratch2/dreamyou070/Prun/result/3_animatelcm_algorithmic_searcher_prun_10_blocks_50_prompts/teacher_folder_result"
    python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path --mode=custom_input    
done
# sbatch -q a100_1_qos -p suma_a100_1 --gres=gpu:1 evaluate_custom.sh
