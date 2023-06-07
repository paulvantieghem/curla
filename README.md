# CURLA
**CURLA: CURL x CARLA**   
Robust end-to-end Autonomous Driving by combining Contrastive Learning and Reinforcement Learning. This repository applies the CURL methodology to an end-to-end autonomous driving task in a custom simulated environment built on top of the CARLA simulator. The code allows users to train and evaluate visual RL agents using the Soft Actor-Critic (SAC) algorithm, combined with an auxiliary self-supervised contrastive learning task (instance discrimination) for better representation learning, leading to increased sample-efficiency and robustness/generalizability.

* CURL: Contrastive Unsupervised Representations for Reinforcement Learning (Laskin et al., 2020) [[Paper](https://arxiv.org/abs/2004.04136)/[Code](https://github.com/MishaLaskin/curl)].
* CARLA: Open-source simulator for autonomous driving research (Dosovitskiy et al., 2017) [[Paper](https://arxiv.org/abs/1711.03938)/[Code](https://github.com/carla-simulator/carla)].

# Table of contents
1. [Installation](#installation)
    1. [CURLA](#curla)
    2. [CARLA](#carla)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Custom Training](#custom_training)
5. [Example results](#example_results)
    1. [Evaluation video](#evaluation_video)
    2. [Visualization of the latent space representations](#latent_viz)


### 1. Installation <a name="installation"></a>
System requirements:
* Linux or Windows operating system
* An NVIDIA GPU with at least 6GB of memory
* At least 30GB of free disk space
* [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Recommended directory structure at the end of the installation:
```
.
├── my_project
│   ├── curla
│   ├── carla
```

#### 1.1 CURLA <a name="curla"></a>
Make the root directory and clone the curla repository:
```
mkdir my_project
cd path/to/my_project
git clone git@github.com:paulvantieghem/curla.git
```

All of the dependencies are in the `environment.yml` file. To create the conda environment, run:
```
cd path/to/my_project/curla
conda env create -f environment.yml
```

#### 1.2 CARLA <a name="carla"></a>
The recommened version of the CARLA simulator is version `0.9.8`, because it is the most recent version proven to be stable enough not to crash during long experiments (~10^6 training steps). More recent versions of the CARLA simulator are compatible (versions `0.9.11` and `0.9.14` were also tested) with the code, but the CARLA simulator is prone to crashing during long training experiments.

Download the CARLA `0.9.8` release and extract it in the `carla` directory:
```
cd path/to/my_project
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.8.tar.gz
tar -xzf CARLA_0.9.8.tar.gz
rm CARLA_0.9.8.tar.gz
```
On Windows, you will have to download and extract the CARLA `0.9.8` release manually. Releases can be downloaded from [the CARLA repository](https://github.com/carla-simulator/carla/releases).

Install the CARLA client library using `conda develop`:
```
conda activate curla
conda install -y conda-build
conda develop path/to/my_project/carla/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
conda deactivate
```

### 2. Training <a name="training"></a>

To train a CURL agent for `N` training steps on the highway of `Town04` with the default (hyper)parameters, run the command listed below. Note that the `train.py` script will automatically launch the CARLA simulator, so no need to launch one manually.
```
conda activate curla
cd path/to/my_project/curla
python train.py --num_train_steps N
```

In your console, you should see printouts that look like:

```
| train | E: 801 | S: 274981 | ER: 344.330 | BR: 0.2396 | A_LOSS: -14.4758 | CR_LOSS: 17.1911 | CU_LOSS: 0.0025
| train | E: 802 | S: 275000 | ER: -6.7408 | BR: 0.0000 | A_LOSS: 0.000000 | CR_LOSS: 0.00000 | CU_LOSS: 0.0000
| eval  | S: 275000 | MER: 104.6457 | BER: 526.8412
| train | E: 803 | S: 276000 | ER: 469.893 | BR: 0.2631 | A_LOSS: -14.9994 | CR_LOSS: 5.94400 | CU_LOSS: 0.0068
| train | E: 804 | S: 277000 | ER: 536.399 | BR: 0.2714 | A_LOSS: -14.4183 | CR_LOSS: 5.31420 | CU_LOSS: 0.0028
```

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

All data related to the run is stored in the specified by `--working_dir_name` (default: `experiments`). To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`.

Tensorboard logging is enabled by default. To view the logs, run the following command in the `curla` directory:
```
tensorboard --logdir experiments --port 6006
```

For custom training, have a look at the section `4. Custom Training`


### 3. Evaluation <a name="evaluation"></a>
At each evaluation step during training, the current CURL model is saved in the `experiments/experiment_name/model` directory. To evaluate a saved model, run:
```
eval.py --experiment_dir_path path/to/experiment_name --model_step 1_000_000
```
Here, `experiment_dir_path` is the path to the directory of the experiment you want to evaluate that contains the `model` directory, and `model_step` is the training step at which the model was saved. The rest of the arguments are automatically read from the `args.log` file in the experiment directory to match the training arguments exactly.


### 4. Custom Training <a name="custom_training"></a>

Possible methods to train agents:
* Pixel SAC: `python train.py --pixel_sac`
* CURL with the identity augmentation: `python train.py --augmentation identity`
* CURL with the random crop augmentation: `python train.py --augmentation random_crop`
* CURL with the color jiggle augmentation: `python train.py --augmentation color_jiggle`
* CURL with the noisy cover augmentation: `python train.py --augmentation noisy_cover`

For hyperparameter tweaking and simulator settings, have a look at the arguments that can be passed to `train.py` and the configuration in `settings.py`. Custom augmentations can be added to `augmentations.py`

Visualization of the augmentations:

<p align="center">
  <img src="https://github.com/paulvantieghem/curla/assets/43028370/360ea7da-ed60-437b-b4c9-3e0c98aa1383" width="600">
</p>

### 5. Example results <a name="example_results"></a>

#### 5.1 Evaluation video <a name="evaluation_video"></a>

https://github.com/paulvantieghem/curla/assets/43028370/e99d6bba-9fdd-4a3f-a4dd-5bb5cd40baf9


**Note**: The oscillations due to sudden steering commands are partly due to the physics in CARLA version `0.9.8`. More recent versions (`>=0.9.10`) have updated turn physics, making the results much smoother. These newer versions of CARLA are not reliable/stable enough for long training however, as will be discuessed below. Another option is add a moving average to the steering command, but this comes at the risk of slower reactions.

#### 5.2 Visualization of the latent space representations <a name="latent_viz"></a>

<p align="center">
  <img src="https://github.com/paulvantieghem/curla/assets/43028370/40e75c63-4db8-441b-b5c5-8e230d32096d" width="800">
</p>

<p align="center">
  <img src="https://github.com/paulvantieghem/curla/assets/43028370/cdd8736f-284f-436e-b810-0d8fe793ca42" width="500">
</p>

