# CURLA
**CURLA: CURL x CARLA**   
Robust end-to-end Autonomous Driving by combining Contrastive Learning and Reinforcement Learning

* [CURL](https://github.com/MishaLaskin/curl): Contrastive Unsupervised Representations for Reinforcement Learning (Laskin et al., 2020).
* [CARLA](https://github.com/carla-simulator/carla): Open-source simulator for autonomous driving research (Dosovitskiy et al., 2017).



### 1. Installation
System requirements:
* Linux or Windows operating system
* A graphics card with at least 4GB of memory
* At least 30GB of free disk space
* [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Recommended directory structure at the end of the installation:
```
.
├── my_project
│   ├── curla
│   ├── carla
```

#### 1.1 CURLA
Make the root directory and clone the curla repository:
```
mkdir my_project
cd path/to/my_project
git clone git@github.com:paulvantieghem/curla.git
```

All of the dependencies are in the `conda_env.yml` file. To create the conda environment, run:
```
cd path/to/my_project/curla
conda env create -f conda_env.yml
```

#### 1.2 CARLA
Download the CARLA 0.9.14 release and extract it in the `carla` directory:
```
cd path/to/my_project
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.14.tar.gz
tar -xzf CARLA_0.9.14.tar.gz
rm CARLA_0.9.14.tar.gz
```
Develop the CARLA egg file in the conda environment
```
conda activate curla
python -m pip install carla==0.9.14
conda deactivate
```

### 2. Training

To train a CURL agent on the highway of `Town04` with the default (hyper)parameters, run:
```
conda activate curla
cd path/to/my_project/curla
python train.py --num_train_steps 1_000_000
```

Tensorboard logging is enabled by default. To view the logs, run the following command in the `curla` directory:
```
tensorboard --logdir tmp --port 6006
```

In your console, you should see printouts that look like:

```
| train | E: 111 | S: 24363 | ER: -21.7607 | BR: -0.086255 | A_LOSS: 4.7732180 | CR_LOSS: 0.1910904 | CU_LOSS: 0.18136342
| train | E: 112 | S: 24840 | ER: -19.5015 | BR: -0.085453 | A_LOSS: 4.8132076 | CR_LOSS: 0.2422579 | CU_LOSS: 0.13720009
| train | E: 113 | S: 25000 | ER: -7.60489 | BR: -0.114820 | A_LOSS: 4.9164285 | CR_LOSS: 0.0465371 | CU_LOSS: 0.10740199
| eval | S: 25000 | MER: -2.6413 | BER: -0.74430
```

For reference, the maximum score for cartpole swing up is around 845 pts, so CURL has converged to the optimal score. This takes about an hour of training depending on your GPU. 

Log abbreviation mapping:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
ER - episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
CU_LOSS - average loss of the CURL encoder

eval - evaluation episode
S - total number of environment steps
MER - mean evaluation episode reward
BER - best evaluation episode reward
```

All data related to the run is stored in the specified by `--working_dir` (default: `tmp`). To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`.

For custom training, have a look at the `train.py` file and the `settings.py` file.

### 3. Evaluation
At each evaluation step during training, the current CURL model is saved in the `tmp/experiment_name/model` directory. To evaluate a saved model, run:
```
eval.py --model_dir_path path/to/model_dir --model_step 1_000_000
```
Here, `model_dir_path` is the path to the directory containing the saved model and `step` is the training step at which the model was saved. More evaluation options are available, have a look at the `eval.py` file for more information. Make sure to match the evaluation settings with the training settings for coherent results.
