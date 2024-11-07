#!/bin/bash

#SBATCH --job-name=animatelcm_finetune_LCM_FeatureMatching
#SBATCH --output=/home/dreamyou070/Prun/logs/animatelcm_finetune_LCM_FeatureMatching.log
#SBATCH --error=/home/dreamyou070/Prun/logs/animatelcm_finetune_LCM_FeatureMatching.log
#SBATCH --time=48:00:00
port_number=55553
# 11 / [0,1,2,3,4,13,14,15,16,17,18]
# 10 / [0,1,2,3,4,13,14,16,17,18]
# 9 / [0,1,2,3,4,13,14,17,18]
# 8 / [0,1,2,3,4,13,14,17]

# find_unused_parameters: true

target_block_num=10
FOLDER="animatelcm_finetune_prun_10_searched_by_latent_openvid_data_simple_transformer_distill_feature_matching"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_1_2_config \
 --main_process_port $port_number \
 distill_with_simple_test.py \
 --wandb_run_name "openvid_data_distill_feature_matching" \
 --output_dir "/scratch2/dreamyou070/Prun/result/Distill/${FOLDER}" \
 --csv_path '/scratch2/dreamyou070/MyData/video/openvid_1M_sample.csv' \
 --video_folder "/scratch2/dreamyou070/MyData/video/openvid_1M/sample" \
 --pretrained_teacher_model '/home/dreamyou070/.cache/huggingface/hub/models--emilianJR--epiCRealism/snapshots/6522cf856b8c8e14638a0aaa7bd89b1b098aed17' \
 --mixed_precision "bf16" \
 --datavideo_size 512 \
 --architecture "[0, 1, 2, 3, 4, 13, 14, 16, 17, 18]" \
 --start_epoch 0 \
 --num_train_epochs 3
