#!/bin/bash

# Load modules
module load python/3.8.2 cudacore/.10.1.243 cuda/10 cudnn/7.6.5

# Create and activate a virtual environment on compute node
virtualenv --no-download env
source env/bin/activate

# Update pip and install Python package
pip install --no-cache-dir --no-index --upgrade pip
pip install --no-cache-dir --no-index torch torchvision
