#!/bin/bash
source /home/.bashrc
exec /home/carla/CarlaUE4.sh -RenderOffScreen -nosound & CARLA=$!
conda activate curla
python -u "$@"
kill $CARLA
kill -9 $(pgrep -f CarlaUE4)
conda deactivate
