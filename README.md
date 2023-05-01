# CURLA
**CURLA: CURL x CARLA**   
Robust end-to-end Autonomous Driving by combining Contrastive Learning and Reinforcement Learning

* CURL: Contrastive Unsupervised Representations for Reinforcement Learning (Laskin et al., 2020) [[Paper](https://arxiv.org/abs/2004.04136)/[Code](https://github.com/MishaLaskin/curl)].
* CARLA: Open-source simulator for autonomous driving research (Dosovitskiy et al., 2017) [[Paper](https://arxiv.org/abs/1711.03938)/[Code](https://github.com/carla-simulator/carla)].



### 1. Installation
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

#### 1.1 CURLA
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

On Windows, you will have to download and extract the CARLA 0.9.14 release manually. Releases can be downloaded from [the CARLA repository](https://github.com/carla-simulator/carla/releases).

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

For custom training, have a look at the `train.py` file and the `settings.py` file.

### 3. Evaluation
At each evaluation step during training, the current CURL model is saved in the `tmp/experiment_name/model` directory. To evaluate a saved model, run:
```
eval.py --model_dir_path path/to/model_dir --model_step 1_000_000
```
Here, `model_dir_path` is the path to the directory containing the saved model and `step` is the training step at which the model was saved. More evaluation options are available, have a look at the `eval.py` file for more information. Make sure to match the evaluation settings with the training settings for coherent results.
