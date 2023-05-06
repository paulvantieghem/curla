#!/bin/bash -l

#SBATCH --time=72:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=train_random_crop
#SBATCH --output=train_random_crop.out
#SBATCH --error=train_random_crop.err

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

# Define the apptainer image, python script, augmentation and log files
IMAGE="$CARLA_ROOT/conda_carla.sif"
SCRIPT="$CONTENT_ROOT/train.py"
AUGMENTATION="random_crop"
LOG_OUT="$LOG_DIR/train_random_crop_$(date +%m-%d_%H-%M).out"
LOG_ERR="$LOG_DIR/train_random_crop_$(date +%m-%d_%H-%M).err"

# Run the apptainer image containing carla and the conda environment to run a python script:
#   * `--nv` binds the NVIDIA drivers such that the GPU can be used from within the apptainer image.
#   * `-B $VSC_HOME` binds the home partitions such that it is visible from within the apptainer image. This is always
#     necessary as python and carla write some cache files there.
#   * `-B $VSC_DATA` and `-B $VSC_SCRATCH` bind the data and scratch partitions such that these are visible from within
#     the apptainer image. These are only necessary if your python script lives there or accesses other files on these
#     partitions.
echo "[MESSAGE] Starting apptainer run"
apptainer run --nv -B $VSC_HOME -B $VSC_DATA -B $VSC_SCRATCH "$IMAGE" "$SCRIPT" "--augmentation" "$AUGMENTATION" > "$LOG_OUT" 2> "$LOG_ERR"
echo "[MESSAGE] Finished apptainer run"
