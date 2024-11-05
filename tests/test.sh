#!/bin/bash

#SBATCH --job-name=down00_automatic_ratio_search
#SBATCH --output=/home/dreamyou070/Prun/logs/20241014_down00_automatic_ratio_search.log
#SBATCH --error=/home/dreamyou070/Prun/logs/20241014_down00_automatic_ratio_search.log
#SBATCH --time=48:00:00

port_number=53609
target_block_num=10
FOLDER="3_animatelcm_algorithmic_searcher_prun_10_blocks_50_prompts_latent/pruned_using_0-1-2-3-4-13-14-16-17-18-architecture"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 test.py \
 --output_dir /scratch2/dreamyou070/Prun/result/${FOLDER} \
 --num_frames 16 \
 --num_inference_step 6 \
 --guidance_scale 1.5 \
 --seed 0 \
 --pretrained_model_path "emilianJR/epiCRealism" --self_attn_pruned --h 512 \
 --architecture "[0, 1, 2, 3, 4, 13, 14, 16, 17, 18]"

# sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 test.sh