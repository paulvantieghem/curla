#!/bin/bash -l

#SBATCH --time=100:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=train_noisy_cover
#SBATCH --output=train_noisy_cover.out
#SBATCH --error=train_noisy_cover.err

# Purge loaded modules
module --force purge

# Load necessary modules
module use /apps/leuven/${VSC_ARCH_LOCAL}/2021a/modules/all
module load intel/2021a
module load libpng
module load libjpeg-turbo
module load CUDA

# Sleep for some time to avoid simulaneous start of differnt jobs
sleep 300

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
AUGMENTATION="noisy_cover"
LOG_OUT="$LOG_DIR/train_noisy_cover_$(date +%m-%d_%H-%M).out"
LOG_ERR="$LOG_DIR/train_noisy_cover_$(date +%m-%d_%H-%M).err"
CARLA_SERVER_PORT=5000
CARLA_TM_PORT=11000

# Search for a range of 3 free ports starting from CARLA_SERVER_PORT
while [ $(netstat -tulpn | grep -E "($CARLA_SERVER_PORT|$((CARLA_SERVER_PORT+1))|$((CARLA_SERVER_PORT+2)))" | wc -l) -ne 0 ]; do
  CARLA_SERVER_PORT=$((CARLA_SERVER_PORT+1))
done
echo "[MESSAGE] Using CARLA_SERVER_PORT=$CARLA_SERVER_PORT"

# Search for a free port starting from CARLA_TM_PORT
while [ $(netstat -tulpn | grep -E $CARLA_TM_PORT | wc -l) -ne 0 ]; do
  CARLA_TM_PORT=$((CARLA_TM_PORT+1))
done
echo "[MESSAGE] Using CARLA_TM_PORT=$CARLA_TM_PORT"

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
