#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1500

# Load modules
module load python/3.8.2 cudacore/.10.1.243 cuda/10 cudnn/7.6.5

# Create variable with the directory for the ML project
SOURCEDIR=~/ml

# Create and activate a virtual environment on compute node
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Update pip and install Python package
pip install --no-index --upgrade pip
pip install --no-index torch torchvision

# Run Python script
python $SOURCEDIR/mlscript.py
