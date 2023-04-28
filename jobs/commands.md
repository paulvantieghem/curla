#### Installations

V100 GPU
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 4 --gpus-per-node=1 --mem-per-cpu=20G -p gpu_v100 jobs/install_carla_img.slurm
```

P100 GPU
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 9 --gpus-per-node=1 --mem-per-cpu=8G -p gpu_p100 jobs/install_carla_img.slurm
```

#### Test
V100 GPU
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 4 --gpus-per-node=1 --mem-per-cpu=20G -p gpu_v100 jobs/test_carla_img.slurm
```

P100 GPU
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 9 --gpus-per-node=1 --mem-per-cpu=8G -p gpu_p100 jobs/test_carla_img.slurm
```

Debug job (max 30 min) P100 GPU
```
sbatch -A lp_edu_alg_parallel_comp -M genius -N 1 -n 9 --gpus-per-node=1 --mem-per-cpu=8G -p gpu_p100 -p batch_debug  jobs/test_carla_img.slurm
```

#### Train
```
sbatch -A lp_rl_thesis -M genius -N 1 -n 4 --gpus-per-node=1 --mem-per-cpu=20G -p gpu_v100 train_carla_img.slurm
```

#### Tensorboard
```
# On your local machine
ssh -L 16006:127.0.0.1:6006 vsc35202@login-genius.hpc.kuleuven.be

# On the server
conda activate curla
cd $VSC_DATA/lib/curla
tensorboard --logdir="tmp" --port=6006

# On your local machine
http://127.0.0.1:16006
```

#### Copying files VSC server <--> local machine
```
# On your local machine: from VSC to local
scp -r vsc35202@login-genius.hpc.kuleuven.be:/data/leuven/352/vsc35202/lib/curla/tmp/ /Users/PaulVT/Desktop/tmp

# On your local machine: from local to VSC
scp -r /Users/PaulVT/Desktop/tmp vsc35202@login-genius.hpc.kuleuven.be:/data/leuven/352/vsc35202/lib/curla
```
