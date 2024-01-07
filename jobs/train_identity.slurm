#!/bin/bash -l

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=train_identity
#SBATCH --output=train_identity.out
#SBATCH --error=train_identity.err

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

# Define the CARLA and content root directories
CARLA_ROOT="$VSC_DATA/lib/carla"
CONTENT_ROOT="$VSC_DATA/lib/curla"

# Create the log directory if it does not exist yet
LOG_DIR="$CONTENT_ROOT/logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# Define the apptainer image, python script, augmentation, log file names and CARLA ports
IMAGE="$CARLA_ROOT/conda_carla.sif"
SCRIPT="$CONTENT_ROOT/train.py"
AUGMENTATION="identity"
LOG_OUT="$LOG_DIR/train_identity_$(date +%m-%d_%H-%M).out"
LOG_ERR="$LOG_DIR/train_identity_$(date +%m-%d_%H-%M).err"
CARLA_SERVER_PORT=2000
CARLA_TM_PORT=8000

# Run the apptainer image containing carla and the conda environment to run a python script:
#   * `--nv` binds the NVIDIA drivers such that the GPU can be used from within the apptainer image.
#   * `-B $VSC_HOME` binds the home partitions such that it is visible from within the apptainer image. This is always
#     necessary as python and carla write some cache files there.
#   * `-B $VSC_DATA` and `-B $VSC_SCRATCH` bind the data and scratch partitions such that these are visible from within
#     the apptainer image. These are only necessary if your python script lives there or accesses other files on these
#     partitions.
echo "[MESSAGE] Starting apptainer run"
apptainer run --nv -B $VSC_HOME -B $VSC_DATA -B $VSC_SCRATCH "$IMAGE" "$SCRIPT" "--augmentation" "$AUGMENTATION" "--server_port" "$CARLA_SERVER_PORT" "--tm_port" "$CARLA_TM_PORT" > "$LOG_OUT" 2> "$LOG_ERR"
echo "[MESSAGE] Finished apptainer run"
