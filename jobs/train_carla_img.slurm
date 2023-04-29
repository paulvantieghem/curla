#!/bin/bash -l

#SBATCH --time=72:00:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=train_carla_img
#SBATCH --output=train_carla_img.out
#SBATCH --error=train_carla_img.err

# Purge loaded modules
module --force purge

# Get the latest version of the curla repository
ssh-add ~/.ssh/id_ed25519
git pull

# Define the CARLA and content root directories
CARLA_ROOT="$VSC_DATA/lib/carla"
CONTENT_ROOT="$VSC_DATA/lib/curla"

# Run the apptainer image containing carla and the conda environment to run a python script:
#   * `--nv` binds the NVIDIA drivers such that the GPU can be used from within the apptainer image.
#   * `-B $VSC_HOME` binds the home partitions such that it is visible from within the apptainer image. This is always
#     necessary as python and carla write some cache files there.
#   * `-B $VSC_DATA` and `-B $VSC_SCRATCH` bind the data and scratch partitions such that these are visible from within
#     the apptainer image. These are only necessary if your python script lives there or accesses other files on these
#     partitions.
#   * `python "$VSC_DATA/carla_test.py"` runs the python script within the activated conda environment of the apptainer image.
apptainer run --nv -B $VSC_HOME -B $VSC_DATA:/vsc_data -B $VSC_SCRATCH "$CARLA_ROOT/conda_carla.sif" python "/vsc_data/lib/curla/train.py"
