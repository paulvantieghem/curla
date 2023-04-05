#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --job-name=curla_installs
#SBATCH --output=curla_installs.out
#SBATCH --error=curla_installs.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=paul.vantieghemdetenberghe@student.kuleuven.be

# Load necessary modules
module --force purge
module use /apps/leuven/icelake/2021a/modules/all
module load intel/2021a
module load libpng
module load libjpeg-turbo
module load CUDA/10.1.105

# Install Miniconda
if [[ -z "$(which conda)" ]]; then
  echo "No installation of conda found, installing Miniconda"
  mkdir -p $VSC_DATA/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $VSC_DATA/miniconda3/miniconda.sh
  bash $VSC_DATA/miniconda3/miniconda.sh -b -u -p $VSC_DATA/miniconda3
  rm -rf $VSC_DATA/miniconda3/miniconda.sh
  $VSC_DATA/miniconda3/bin/conda init bash
  $VSC_DATA/miniconda3/bin/conda init zsh
  export CONDA_ALWAYS_YES="true" # Always answer yes to conda prompts
else
  echo "Found conda installation, skipping installation of Miniconda"
fi

# Make necessary directories
mkdir -p $VSC_DATA/lib/carla

# Clone the CURLA repository and install the conda environment
if [ -d "$VSC_DATA/lib/curla" ]; then
  rm -rf $VSC_DATA/lib/curla
fi
cd $VSC_DATA/lib/
git clone https://github.com/paulvantieghem/curla.git
cd $VSC_DATA/lib/curla
conda env create -f conda_env.yml
conda activate curla

# Install CARLA 0.9.11
cd $VSC_DATA/lib/carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
tar -xzf CARLA_0.9.11.tar.gz
rm CARLA_0.9.11.tar.gz
conda activate curla
export CONDA_ALWAYS_YES="true"
conda install conda-build
conda develop ./PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg

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
