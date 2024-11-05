#!/bin/bash

#SBATCH --job-name=down00_automatic_ratio_search
#SBATCH --output=/home/dreamyou070/Prun/logs/20241014_down00_automatic_ratio_search.log
#SBATCH --error=/home/dreamyou070/Prun/logs/20241014_down00_automatic_ratio_search.log
#SBATCH --time=48:00:00

port_number=53611
FOLDER="20241026_feature_reuse_and_skip_polity"
skip_layer='up_blocks_3_motion_modules_2'
target_time=2
accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 motion_feature_reuse.py \
 --output_dir /scratch2/dreamyou070/Prun/result/${FOLDER} \
 --num_frames 16 \
 --num_inference_step 6 \
 --guidance_scale 1.5 \
 --seed 0 \
 --pretrained_model_path "emilianJR/epiCRealism" --self_attn_pruned --h 512 \
 --skip_layers "['${skip_layer}']" \
 --skip_layers_dot "['up_blocks.3.motion_modules.2']"
# sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 test.sh
