#!/bin/bash
#SBATCH --job-name=twelve
#SBATCH -t 2-00:00              		# Runtime in D-HH:MM
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

module load anaconda3
conda activate ICE_NN
python3 2dCNN.py
