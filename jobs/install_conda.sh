#!/bin/bash

# Install Miniconda
mkdir -p $VSC_DATA/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $VSC_DATA/miniconda3/miniconda.sh
bash $VSC_DATA/miniconda3/miniconda.sh -b -u -p $VSC_DATA/miniconda3
rm -rf $VSC_DATA/miniconda3/miniconda.sh
$VSC_DATA/miniconda3/bin/conda init bash
$VSC_DATA/miniconda3/bin/conda init zsh
export CONDA_ALWAYS_YES="true" # Always answer yes to conda prompts
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh # Export to be made available in subshells