#!/bin/bash
#SBATCH --job-name=CNN-train
#SBATCH --output=/home/mgatlin3/FT/logs/out/%x_%j.out
#SBATCH --error=/home/mgatlin3/FT/logs/err/%x_%j.err
#SBATCH --export=NONE
#SBATCH --mem=128G
#SBATCH -n 2
#SBATCH -N 2
#SBATCH -t 0-8:00
#SBATCH -c 16
#SBATCH --gres=gpu:2


module load anaconda3
conda activate poop
cd /home/mgatlin3/FT/
python 2dCNN.py