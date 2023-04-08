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