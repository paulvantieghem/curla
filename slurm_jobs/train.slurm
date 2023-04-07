#!/bin/bash

#SBATCH --time=10:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=curla_train
#SBATCH --output=curla_train.out
#SBATCH --error=curla_train.err

# Exit on error
set -e

# Base modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load anaconda3/2022.10
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

# Install Miniconda
if [[ -z "$(which conda)" ]]; then
  echo "[MESSAGE] No installation of conda found afte setting environment variables, exiting..."
  exit 1
fi

# Clone the CURLA repository
if [ -d "$VSC_DATA/lib/curla" ]; then
  rm -rf $VSC_DATA/lib/curla
fi
cd $VSC_DATA/lib/
git clone https://github.com/paulvantieghem/curla.git

# Launch the CARLA simulator
cd $VSC_DATA/lib/carla
./CarlaUE4.sh & CARLA=$!

# Train CURL agent on CARLA simulator
cd $VSC_DATA/lib/curla
conda activate curla
python train.py --num_train_steps 200000

# Terminate session
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
conda deactivate
