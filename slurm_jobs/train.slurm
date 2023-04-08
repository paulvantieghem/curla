#!/bin/bash

#SBATCH --time=12:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=curla_train
#SBATCH --output=curla_train.out
#SBATCH --error=curla_train.err

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
  echo "[MESSAGE] No installation of conda found afte setting environment variables, exiting..."
  exit 1
fi

# Refresh the curla repository
if [ -d "$VSC_DATA/lib/curla" ]; then
  rm -rf $VSC_DATA/lib/curla
fi
cd $VSC_DATA/lib/
git clone https://github.com/paulvantieghem/curla.git

# Launch the CARLA simulator in the background
cd $VSC_DATA/lib/carla
DISPLAY=
SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl & CARLA=$!

# Sleep for 30 seconds
sleep 30

# Train CURL agent on CARLA simulator
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
conda activate curla
cd $VSC_DATA/lib/curla
python train.py --num_train_steps 200000

# Terminate session
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
conda deactivate
