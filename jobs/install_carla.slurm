#!/bin/bash

#SBATCH --time=02:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=curla_install
#SBATCH --output=curla_install.out
#SBATCH --error=curla_install.err

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

# Define the CARLA and content root directories
CARLA_ROOT="$VSC_DATA/lib/carla"

# Install conda environment
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
export CONDA_ALWAYS_YES="true"
cd $VSC_DATA/lib/curla
conda env create -f environment_2.yml
conda activate curla
conda install conda-build
cd $VSC_DATA/lib/carla
conda develop ./PythonAPI/carla/dist/carla-0.9.8-py3.7-linux-x86_64.egg
conda deactivate
echo "[MESSAGE] Conda environment installed"