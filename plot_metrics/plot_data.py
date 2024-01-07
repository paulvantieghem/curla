# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Constants
PLOT_DIR = './plots'
EXP_DIR = '../experiments'
EXPERIMENTS = ['pixel_sac', 'identity', 'random_crop', 'color_jiggle', 'noisy_cover', 'identity_detached']

# Define the metrics you want to plot
metrics = ['train/ep_reward', 'train/batch_reward', 'train_critic/loss', 'eval/mean_ep_reward', 'eval/mean_ep_steps', 'eval/z_mean_ep_mean_kmh']

# Custom mean function
def my_mean(arr):
    if len(arr) == 0:
        return 0.0
    else:
        return np.mean(arr)

# Initialize the data dictionary
data_dict = {experiment: {} for experiment in EXPERIMENTS}

# Get the list of experiment directories
experiments = os.listdir(EXP_DIR)
experiments = [exp for exp in experiments if os.path.isdir(os.path.join(EXP_DIR, exp))]
experiments.sort()
for exp in experiments:
    print(exp)

# Add the metrics to the data dictionary
for exp_type in data_dict:
    for metric in metrics:
        data_dict[exp_type][metric] = {'individual': [], 'mean': None, 'max': None, 'min': None}

# Iterate over the experiments
for log_dir in tqdm(experiments):

    # Get the type of the experiment
    exp_type = log_dir.split('-')[-1]

    # Load the event accumulator
    event_acc = EventAccumulator(os.path.join(EXP_DIR, log_dir, 'tb'))
    event_acc.Reload()

    # Iterate over the metrics
    for metric in metrics:
        x, y = [], []
        for scalar_event in event_acc.Scalars(metric):
            x.append(scalar_event.step)
            y.append(scalar_event.value)
        x, y = np.array(x), np.array(y)

        # Discretize the x,y values per 1000 steps, take mean if multiple values
        if 'train' in metric:
            new_x = np.arange(0, 1e6 + 1e3, 1e3)
            new_y = np.zeros_like(new_x)
            for i in range(1, len(new_x)):
                new_y[i] = my_mean(y[np.logical_and(x >= new_x[i] - 1e3, x < new_x[i])])
            y = new_y
        data_dict[exp_type][metric]['individual'].append(y)
    
# Compute the mean and std
for exp_type in data_dict:
    for metric in metrics:
        data_dict[exp_type][metric]['individual'] = np.array(data_dict[exp_type][metric]['individual'])
        data_dict[exp_type][metric]['mean'] = np.mean(data_dict[exp_type][metric]['individual'], axis=0)
        data_dict[exp_type][metric]['max'] = np.max(data_dict[exp_type][metric]['individual'], axis=0)
        data_dict[exp_type][metric]['min'] = np.min(data_dict[exp_type][metric]['individual'], axis=0)

# Save the data dictionary
np.save('./data_dict.npy', data_dict)

# Load the data dictionary
data_dict = np.load('./data_dict.npy', allow_pickle=True).item()