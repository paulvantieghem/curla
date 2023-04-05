#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --job-name=curla_train
#SBATCH --output=curla_train.out
#SBATCH --error=curla_train.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

# Exit on error
set -e

# Base modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load anaconda3/2022.10
module load libpng
module load libjpeg-turbo

if [[ -z "$(which conda)" ]]; then
  echo "No installation of conda found!"
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
python train.py --num_train_steps 100000

# Terminate session
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
conda deactivate
