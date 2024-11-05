#!/bin/bash

#SBATCH --job-name=distill_loss
#SBATCH --output=/home/dreamyou070/Prun/logs/20241014_distill_loss.log
#SBATCH --error=/home/dreamyou070/Prun/logs/20241014_distill_loss.log
#SBATCH --time=48:00:00

port_number=50001
FOLDER="blockwise_finding_compare_with_teacher_distill_loss_teacher_gen_data"

accelerate launch \
 --config_file $HOME/gpu_config/gpu_0_config \
 --main_process_port $port_number \
 blockwise_finding.py \
 --output_dir /scratch2/dreamyou070/Prun/result/2_FVD_substitute/${FOLDER} \
 --num_frames 16 \
 --num_inference_step 6 \
 --guidance_scale 1.5 \
 --seed 0 \
 --datavideo_size 512 \
 --pretrained_model_path "emilianJR/epiCRealism" --self_attn_pruned --h 512 \
 --csv_path = r'/scratch2/dreamyou070/Prun/result/teacher_calibration_data/sample.csv'
 --video_folder = r'/scratch2/dreamyou070/Prun/result/teacher_calibration_data/sample'
 # sbatch -p suma_a100_1 -q a100_1_qos --gres=gpu:1 blockwise_finding.sh