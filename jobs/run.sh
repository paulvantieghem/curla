#!/bin/bash
source /home/.bashrc

echo "[$(date +%H:%M:%S)] Starting training"
conda activate curla
python -u "$@"
conda deactivate
echo "[$(date +%H:%M:%S)] Training finished"
