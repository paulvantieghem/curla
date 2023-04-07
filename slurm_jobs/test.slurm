#!/bin/bash
#SBATCH --cluster=genius
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G

#SBATCH --time=00:30:00

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

#SBATCH --job-name=curla_test
#SBATCH --output=curla_test.out
#SBATCH --error=curla_test.err

# Load necessary modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load libpng
module load libjpeg-turbo
module load CUDA/10.1.105

# Launch the CARLA simulator in the background
cd $VSC_DATA/lib/carla
./CarlaUE4.sh & CARLA=$!

# Sleep for 30 seconds
sleep 30

# Run test script
python $VSC_DATA/lib/curla/test.py

# Terminate session
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
conda deactivate
