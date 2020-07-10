#!/bin/bash
#SBATCH --time=0:5:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Activate your virtual env
source ~/env/bin/activate

# Run your Python script
python ~/mnist/cnn.py
