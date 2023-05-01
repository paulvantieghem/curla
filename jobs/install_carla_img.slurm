#!/bin/bash -l

#SBATCH --time=01:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=install_carla_img
#SBATCH --output=install_carla_img.out
#SBATCH --error=install_carla_img.err

# Purge loaded modules
module --force purge
module load CUDA

# Get the latest version of the curla repository
ssh-add ~/.ssh/id_ed25519
git pull

# Create CARLA root directory
CARLA_ROOT="$VSC_DATA/lib/carla"
CONTENT_ROOT="$VSC_DATA/lib/curla"
mkdir -p $CARLA_ROOT

export APPTAINER_TMPDIR="$VSC_SCRATCH/apptainer/tmp"
export APPTAINER_CACHEDIR="$VSC_SCRATCH/apptainer/cache"
mkdir -p $APPTAINER_TMPDIR
mkdir -p $APPTAINER_CACHEDIR

# Remove the .sif file if it already exists
if [ -f $CARLA_ROOT/conda_carla.sif ]; then
  rm -f $CARLA_ROOT/conda_carla.sif
fi

# Build the apptainer image containing carla and the conda environment:
echo "[MESSAGE] Starting apptainer build"
apptainer build --nv "$CARLA_ROOT/conda_carla.sif" ./jobs/conda_carla.def > "$CONTENT_ROOT/install_apptainer_build_$(date +%m-%d_%H-%M).out" 2> "$CONTENT_ROOT/install_apptainer_build_$(date +%m-%d_%H-%M).err"
echo "[MESSAGE] Finished apptainer build"

# Run the apptainer image containing carla and the conda environment to test if the python script runs:
echo "[MESSAGE] Starting apptainer run"
apptainer run --nv -B $VSC_HOME -B $VSC_DATA -B $VSC_SCRATCH "$CARLA_ROOT/conda_carla.sif" python "$CONTENT_ROOT/train.py" > "$CONTENT_ROOT/install_apptainer_run_$(date +%m-%d_%H-%M).out" 2> "$CONTENT_ROOT/install_apptainer_run_$(date +%m-%d_%H-%M).err"
echo "[MESSAGE] Finished apptainer run"
