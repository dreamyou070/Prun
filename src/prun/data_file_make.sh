#!/bin/bash

#SBATCH --job-name=data_file_make
#SBATCH --output=/home/dreamyou070/Prun/logs/data_file_make.log
#SBATCH --error=/home/dreamyou070/Prun/logs/data_file_make.log
#SBATCH --time=48:00:00

python data_file_make.py

# sbatch -p base_suma_rtx3090 -q big_qos --gres=gpu:1 --time 48:00:00 data_file_make.sh