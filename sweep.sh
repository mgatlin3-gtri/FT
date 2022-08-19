#!/bin/bash
#SBATCH --job-name=CNN-super-sweep
#SBATCH --output=/home/mgatlin3/FT/logs/out/%x_%j.out
#SBATCH --error=/home/mgatlin3/FT/logs/err/%x_%j.err
#SBATCH --export=NONE
#SBATCH --mem=256G
#SBATCH -n 3
#SBATCH -N 3
#SBATCH -t 3-12:00
#SBATCH -c 32
#SBATCH --gres=gpu:3 


module load anaconda3
module load gcc/9.2.0
conda activate poop
cd /home/mgatlin3/FT
python sweep.py