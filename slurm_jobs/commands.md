#### Installations
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 4 --gpus-per-node=1 --mem-per-cpu=20G -p gpu_v100 install_conda.slurm
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 4 --gpus-per-node=1 --mem-per-cpu=20G -p gpu_v100 install_curla.slurm
```

#### Test
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 4 --gpus-per-node=1 --mem-per-cpu=20G -p gpu_v100 test.slurm
```


#### Train
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 4 --gpus-per-node=1 --mem-per-cpu=20G -p gpu_v100 train.slurm
```

#### Tensorboard
```
# On your local machine
ssh -L 16006:127.0.0.1:6006 vsc35202@login-genius.hpc.kuleuven.be

# On the server
cd $VSC_DATA/lib/curla
tensorboard --logdir="tmp" --port=6006

# On your local machine
http://127.0.0.1:16006
```