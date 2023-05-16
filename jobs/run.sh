#!/bin/bash
source /home/.bashrc

# Check if CARLA_SERVER_PORT is free
for port in "$5" "$(( $5 + 1 ))" "$(( $5 + 2 ))"; do
    if lsof -i ":$port" >/dev/null 2>&1; then
        echo "Port $port is in use by the following process(es):"
        lsof -i ":$port" | awk 'NR>1 {print "  PID:", $2, "Process name:", $1}'
    else
        echo "Port $port is free."
    fi
done

# Check if CARLA_TM_PORT is free
for port in "$7"; do
    if lsof -i ":$port" >/dev/null 2>&1; then
        echo "Port $port is in use by the following process(es):"
        lsof -i ":$port" | awk 'NR>1 {print "  PID:", $2, "Process name:", $1}'
    else
        echo "Port $port is free."
    fi
done

echo "[$(date +%H:%M:%S)] Starting training"
conda activate curla
python -u "$@"
conda deactivate
echo "[$(date +%H:%M:%S)] Training finished"
