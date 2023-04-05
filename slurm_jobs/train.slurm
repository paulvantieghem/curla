#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --job-name="CURLA_v0"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=32G
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

# Base modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load anaconda3/2022.10
module load libpng
module load libjpeg-turbo

# Set the CONDA_ALWAYS_YES environment variable
export CONDA_ALWAYS_YES="true"

# Launch the CARLA simulator
cd $VSC_DATA/lib/carla
./CarlaUE4.sh & CARLA=$!

# Train CURL agent on CARLA simulator
cd $VSC_DATA/lib/curla
conda activate curla
python train.py --num_train_steps 100000

# Terminate session
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
conda deactivate
