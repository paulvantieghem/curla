#!/bin/bash
source /home/.bashrc

echo "[$(date +%H:%M:%S)] Starting CARLA"
exec /home/carla/CarlaUE4.sh -RenderOffScreen -nosound & CARLA=$!

echo "[$(date +%H:%M:%S)] Sleeping for 60 seconds"
sleep 60

echo "[$(date +%H:%M:%S)] Starting training"
conda activate curla
python -u "$@"
conda deactivate

echo "[$(date +%H:%M:%S)] Training finished, killing CARLA processes"
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
