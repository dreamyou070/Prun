#!/bin/bash

#SBATCH --job-name=down21_automatic_ratio_search
#SBATCH --output=/home/dreamyou070/Prun/logs/20241014_down21_automatic_ratio_search.log
#SBATCH --error=/home/dreamyou070/Prun/logs/20241014_down21_automatic_ratio_search.log
#SBATCH --time=48:00:00

port_number=52002
FOLDER="20241017_layerwise_sorting_3_total"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 test_total_layerwise_inference.py \
 --output_dir /scratch2/dreamyou070/Prun/result/${FOLDER} \
 --sub_folder_name "down00_down01_down10" \
 --num_frames 16 \
 --num_inference_step 6 \
 --guidance_scale 1.5 \
 --seed 0 \
 --pretrained_model_path "emilianJR/epiCRealism" --self_attn_pruned --h 512 \
 --skip_layers "['up_blocks_0_motion_modules_2']" \
 --skip_layers_dot "['up_blocks.0.motion_modules.2']" \
 --pruning_ratio_list "[]"
# sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 timewise_permutation.sh

