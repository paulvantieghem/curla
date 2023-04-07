#!/bin/bash

#SBATCH --time=00:30:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=curla_install
#SBATCH --output=curla_install.out
#SBATCH --error=curla_install.err

# Load necessary modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load libpng
module load libjpeg-turbo
module load CUDA/11.7

# CUDA version
nvcc --version

# Set environment variables
if [[ -z "$(which conda)" ]]; then
  echo "[MESSAGE] No installation of conda found, adding miniconda3 to PATH"
  export PATH="$VSC_DATA/miniconda3/bin:$PATH"
  $VSC_DATA/miniconda3/bin/conda init bash
  $VSC_DATA/miniconda3/bin/conda init zsh
fi

# Check if conda is installed
if [[ -z "$(which conda)" ]]; then
  echo "[MESSAGE] No installation of conda found afte setting environment variables, exiting..."
  exit 1
fi

# Make necessary directories
mkdir -p $VSC_DATA/lib/carla

# Clone the CURLA repository and install the conda environment
if [ -d "$VSC_DATA/lib/curla" ]; then
  rm -rf $VSC_DATA/lib/curla
fi
cd $VSC_DATA/lib/
git clone https://github.com/paulvantieghem/curla.git
cd $VSC_DATA/lib/curla
conda env create -f conda_env.yml
echo "[MESSAGE] Conda environment 'curla' created"

# Install CARLA 0.9.11
cd $VSC_DATA/lib/carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
tar -xzf CARLA_0.9.11.tar.gz
rm CARLA_0.9.11.tar.gz
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
export CONDA_ALWAYS_YES="true"
conda activate curla
conda install conda-build
conda develop ./PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
conda deactivate
echo "[MESSAGE] CARLA installed"
