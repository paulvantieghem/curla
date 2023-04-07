#!/bin/bash

#SBATCH --time=00:30:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=conda_install
#SBATCH --output=conda_install.out
#SBATCH --error=conda_install.err

# Load necessary modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load libpng
module load libjpeg-turbo
module load CUDA/11.7

# CUDA version
nvcc --version

# Install Miniconda
mkdir -p $VSC_DATA/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $VSC_DATA/miniconda3/miniconda.sh
bash $VSC_DATA/miniconda3/miniconda.sh -b -u -p $VSC_DATA/miniconda3
rm -rf $VSC_DATA/miniconda3/miniconda.sh
$VSC_DATA/miniconda3/bin/conda init bash
$VSC_DATA/miniconda3/bin/conda init zsh
export CONDA_ALWAYS_YES="true" # Always answer yes to conda prompts
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh # Export to be made available in subshells
