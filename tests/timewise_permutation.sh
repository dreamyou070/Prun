#!/bin/bash

#SBATCH --job-name=up02_timewise_permutation
#SBATCH --output=/home/dreamyou070/Prun/logs/20241014_up02_timewise_permutation.log
#SBATCH --error=/home/dreamyou070/Prun/logs/20241014_up02_timewise_permutation.log
#SBATCH --time=48:00:00

port_number=50112
FOLDER="20241025_layerwise_pruning"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 timewise_permutation.py \
 --output_dir /scratch2/dreamyou070/Prun/result/${FOLDER} \
 --num_frames 16 \
 --num_inference_step 6 \
 --guidance_scale 1.5 \
 --seed 0 \
 --pretrained_model_path "emilianJR/epiCRealism" --self_attn_pruned --h 512 \
 --skip_layers "['down_blocks_0_motion_modules_0']" \
 --skip_layers_dot "['down_blocks.0.motion_modules.0']" \
 --pruning_ratio_list "[]" \
 --target_timestep 1
# sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 timewise_permutation.sh
