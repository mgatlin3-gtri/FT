#!/bin/bash
#SBATCH --job-name=CNN-train
#SBATCH --output=/home/mgatlin3/FT/logs/out/%x_%j.out
#SBATCH --error=/home/mgatlin3/FT/logs/err/%x_%j.err
#SBATCH --export=NONE
#SBATCH --mem=128G
#SBATCH -n 3
#SBATCH -N 3
#SBATCH -t 0-16:00
#SBATCH -c 16
#SBATCH --gres=gpu:3 


module load anaconda3
module load gcc/9.2.0
conda activate poop
cd /home/mgatlin3/FT/
python 2dCNN.py