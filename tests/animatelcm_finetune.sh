#!/bin/bash

#SBATCH --job-name=animatelcm_finetune_noprun
#SBATCH --output=/home/dreamyou070/Prun/logs/animatelcm_finetune_noprun.log
#SBATCH --error=/home/dreamyou070/Prun/logs/animatelcm_finetune_noprun.log
#SBATCH --time=48:00:00
port_number=52220
# 11 / [0,1,2,3,4,13,14,15,16,17,18]
# 10 / [0,1,2,3,4,13,14,16,17,18]
# 9 / [0,1,2,3,4,13,14,17,18]
# 8 / [0,1,2,3,4,13,14,17]

# find_unused_parameters: true
target_block_num=10
FOLDER="4_animatelcm_finetune_prun_${target_block_num}_panda_data_register_code_testing"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 animatelcm_finetune_3.py \
 --output_dir /scratch2/dreamyou070/Prun/result/${FOLDER} \
 --num_frames 16 \
 --num_inference_step 6 \
 --guidance_scale 1.5 \
 --seed 0 \
 --architecture "[0,1,2,3,4,6,14,15,16,20]" \
 --pretrained_teacher_model "emilianJR/epiCRealism" \
 --num_train_epochs 30000 \
 --train_batch_size 3 \
 --csv_path /scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv \
 --video_folder /scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample

# sbatch -p suma_a6000 -q big_qos --gres=gpu:1 --time 48:00:00 animatelcm_evolutionary_algorithm_search.sh
# sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 --time 48:00:00 animatelcm_finetune.sh
