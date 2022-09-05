#!/bin/bash
#SBATCH --job-name=smallCNN-sweep
#SBATCH --output=/home/mgatlin3/FT/logs/out/%x_%j.out
#SBATCH --error=/home/mgatlin3/FT/logs/err/%x_%j.err
#SBATCH --export=NONE
#SBATCH --mem=256G
#SBATCH -n 4
#SBATCH -N 4
#SBATCH -t 5-12:00
#SBATCH -c 32
#SBATCH --gres=gpu:3 


module load anaconda3
module load gcc/9.2.0
conda activate poop
cd /home/mgatlin3/FT
python sweep-smallerCNN.py