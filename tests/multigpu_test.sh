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
 --config_file $HOME/gpu_config/gpu_0_1_config \
 --main_process_port $port_number \
 multigpu_test.py \
