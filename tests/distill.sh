#!/bin/bash

#SBATCH --job-name=animatelcm_finetune_LCM_FeatureMatching
#SBATCH --output=/home/dreamyou070/Prun/logs/animatelcm_finetune_LCM_FeatureMatching.log
#SBATCH --error=/home/dreamyou070/Prun/logs/animatelcm_finetune_LCM_FeatureMatching.log
#SBATCH --time=48:00:00
port_number=55555
# 11 / [0,1,2,3,4,13,14,15,16,17,18]
# 10 / [0,1,2,3,4,13,14,16,17,18]
# 9 / [0,1,2,3,4,13,14,17,18]
# 8 / [0,1,2,3,4,13,14,17]

# find_unused_parameters: true
target_block_num=10
FOLDER="multigpu_test"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 distill.py \
 --output_dir "/scratch2/dreamyou070/Prun/result/distill/${FOLDER}" \
 --use_8bit_adam \
 --mixed_precision no \
 --csv_path "/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv" \
 --video_folder "/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample" \
 --per_gpu_batch_size 1 \
 --gradient_checkpointing \
 --datavideo_size 512 \
 --architecture "[0, 1, 2, 3, 4, 13, 14, 16, 17, 18]"


# sbatch -p suma_a6000 -q big_qos --gres=gpu:1 --time 48:00:00 animatelcm_evolutionary_algorithm_search.sh
# sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 --time 96:00:00 distill.sh

#--csv_path "/scratch2/dreamyou070/MyData/video/openvid_1M/sample.csv" \
 #--video_folder "/scratch2/dreamyou070/MyData/video/openvid_1M"
