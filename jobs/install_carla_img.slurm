#!/bin/bash -l

#SBATCH --time=02:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=install_carla_img
#SBATCH --output=install_carla_img.out
#SBATCH --error=install_carla_img.err

# Purge loaded modules
module --force purge

# Load necessary modules
module use /apps/leuven/${VSC_ARCH_LOCAL}/2021a/modules/all
module load intel/2021a
module load libpng
module load libjpeg-turbo
module load CUDA

# Check CUDA version
echo "nvcc --version:"
nvcc --version
echo "nvidia-smi:"
nvidia-smi

# Get the latest version of the curla repository
ssh-add ~/.ssh/id_ed25519
git pull

# Create CARLA root directory
CARLA_ROOT="$VSC_DATA/lib/carla"
mkdir -p $CARLA_ROOT

export APPTAINER_TMPDIR="$VSC_SCRATCH/apptainer/tmp"
export APPTAINER_CACHEDIR="$VSC_SCRATCH/apptainer/cache"
mkdir -p $APPTAINER_TMPDIR
mkdir -p $APPTAINER_CACHEDIR

# Remove the .sif file if it already exists
if [ -f $CARLA_ROOT/conda_carla.sif ]; then
  rm -f $CARLA_ROOT/conda_carla.sif
fi

apptainer build --nv "$CARLA_ROOT/conda_carla.sif" ./jobs/conda_carla.def