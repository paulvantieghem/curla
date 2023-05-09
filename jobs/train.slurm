#!/bin/bash

#SBATCH --time=72:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=train_curla
#SBATCH --output=train_curla.out
#SBATCH --error=train_curla.err

# Load necessary modules
module --force purge
module use /apps/leuven/${VSC_ARCH_LOCAL}/2021a/modules/all
module load intel/2021a
module load libpng
module load libjpeg-turbo
module load CUDA

# CUDA version
nvcc --version

# GPU information 
nvidia-smi

# Set environment variables
if [[ -z "$(which conda)" ]]; then
  echo "[MESSAGE] No installation of conda found, adding miniconda3 to PATH"
  export PATH="$VSC_DATA/miniconda3/bin:$PATH"
  $VSC_DATA/miniconda3/bin/conda init bash
  $VSC_DATA/miniconda3/bin/conda init zsh
fi

# Check if conda is installed
if [[ -z "$(which conda)" ]]; then
  echo "[MESSAGE] No installation of conda found after setting environment variables, exiting..."
  exit 1
fi

# Go to curla directory
cd $VSC_DATA/lib/curla

# Get the latest version of the curla repository
ssh-add ~/.ssh/id_ed25519
git pull

# Define the CARLA and content root directories
CARLA_ROOT="$VSC_DATA/lib/carla"

# Install conda environment
conda activate curla
python train.py
conda deactivate