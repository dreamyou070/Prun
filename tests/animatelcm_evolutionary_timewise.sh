#!/bin/bash

#SBATCH --job-name=block_18_11
#SBATCH --output=/home/dreamyou070/Prun/logs/block_18_11.log
#SBATCH --error=/home/dreamyou070/Prun/logs/block_18_11.log
#SBATCH --time=48:00:00
port_number=53333
# 11 / [0,1,2,3,4,13,14,15,16,17,18]
# 10 / [0,1,2,3,4,13,14,16,17,18]
# 9 / [0,1,2,3,4,13,14,17,18]
# 8 / [0,1,2,3,4,13,14,17]


target_block_num=10
FOLDER="3_animatelcm_algorithmic_searcher_18_11_blocks_50_prompts"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 animatelcm_evolutionary_timewise.py \
 --output_dir /scratch2/dreamyou070/Prun/result/${FOLDER} \
 --num_frames 16 \
 --num_inference_step 6 \
 --guidance_scale 1.5 \
 --seed 0 \
 --target_block_num ${target_block_num} \
 --pretrained_model_path "emilianJR/epiCRealism" \
 --init_architecture "[0,1,2,3,4,13,14,16,17,18]" \
 --population_size 6 --mutation_num 2 --max_prompt 50


# sbatch -p suma_a6000 -q big_qos --gres=gpu:1 --time 48:00:00 animatelcm_evolutionary_timewise.sh
# sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 --time 48:00:00 animatelcm_evolutionary_algorithm_search.sh