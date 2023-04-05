#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --job-name="CURLA_installs"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

# Base modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load anaconda3/2022.10
module load libpng
module load libjpeg-turbo

# Clone the CURLA repo and install conda environment
cd $VSC_DATA/lib/
git clone git@github.com:paulvantieghem/curla.git
cd $VSC_DATA/lib/curla
echo "[PAUL] Curla repository cloned"
conda env create -y -f conda_env.yml
conda activate curla
echo "[PAUL] Curla conda environment created"

# Install CARLA 0.9.11
cd $VSC_DATA/lib/
mkdir -p $VSC_DATA/lib/carla
cd $VSC_DATA/lib/carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
tar -xzf CARLA_0.9.11.tar.gz
rm CARLA_0.9.11.tar.gz
conda activate curla
conda install -y conda-build
conda develop ./PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
echo "[PAUL] Carla 0.9.11 installed"

# Launch the CARLA simulator in the background
cd $VSC_DATA/lib/carla
./CarlaUE4.sh & CARLA=$!

# Sleep for 30 seconds
sleep 30

# Terminate session
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
conda deactivate
