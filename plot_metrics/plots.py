# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the data dictionary
data_dict = np.load('./data_dict.npy', allow_pickle=True).item()
experiments = list(data_dict.keys())
metrics = ['train/ep_reward', 'train/batch_reward', 'train_critic/loss', 'eval/mean_ep_reward', 'eval/mean_ep_steps', 'eval/z_mean_ep_mean_kmh']

def get_label(key):
    if key == 'pixel_sac':
        return 'Pixel SAC'
    elif key == 'identity':
        return 'CURL identity'
    elif key == 'random_crop':
        return 'CURL random crop'
    elif key == 'color_jiggle':
        return 'CURL color jiggle'
    elif key == 'noisy_cover':
        return 'CURL noisy cover'
    elif key == 'identity_detached':
        return 'CURL identity detached'
    
def get_y_label(metric):
    if metric == 'eval/mean_ep_reward':
        return 'Mean Episode Return'
    elif metric == 'eval/mean_ep_steps':
        return 'Mean Episode Steps'
    elif metric == 'train_critic/loss':
        return 'Critic Loss'
    elif metric == 'train/ep_reward':
        return 'Episode Return'
    elif metric == 'train/batch_reward':
        return 'Mean Batch Reward'
    elif metric == 'eval/z_mean_ep_mean_kmh':
        return 'Mean Episode Mean Speed [km/h]'

def exponential_smoothing(data, alpha):
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # Initial forecast

    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t-1] + (1 - alpha) * smoothed_data[t-1]

    return smoothed_data

def moving_average_smoothing(data, window_size):
    smoothed_data = np.append(data, np.repeat(np.mean(data[-window_size:]), window_size-1))
    smoothed_data = np.convolve(smoothed_data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data
    
# Get list of colors in 'Set1' colormap
colors = sns.color_palette('Set1', n_colors=9)

# Define smoothing type
smoothing = 'moving_average'

# Plot all experiments per metric, but smooth the data first
for metric in metrics:
    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    for i, key in enumerate(experiments):
        data = data_dict[key][metric]['mean']
        min_data = data_dict[key][metric]['min']
        max_data = data_dict[key][metric]['max']
        if 'train' in metric:
            if smoothing == 'moving_average':
                window_size = 100
                data = moving_average_smoothing(data, window_size)
                min_data = moving_average_smoothing(min_data, window_size)
                max_data = moving_average_smoothing(max_data, window_size)
            else:
                alpha = 0.1
                data = exponential_smoothing(data, alpha)
                min_data = exponential_smoothing(min_data, alpha)
                max_data = exponential_smoothing(max_data, alpha)
            steps = np.arange(0, 1e6 + 1e3, 1e3)[:len(data)]
        elif 'eval' in metric:
            if smoothing == 'moving_average':
                window_size = 10
                data = moving_average_smoothing(data, window_size)
                min_data = moving_average_smoothing(min_data, window_size)
                max_data = moving_average_smoothing(max_data, window_size)
            else:
                alpha = 0.3
                data = exponential_smoothing(data, alpha)
                min_data = exponential_smoothing(min_data, alpha)
                max_data = exponential_smoothing(max_data, alpha)
            steps = np.arange(0, 1e6 + 25*1e3, 25*1e3)[:len(data)]
        plt.plot(steps, data, label=get_label(key), color=colors[i])
        plt.fill_between(steps, min_data, max_data, alpha=0.2, color=colors[i])

    # Add target speed line
    if metric == 'eval/z_mean_ep_mean_kmh':
        plt.axhline(y=65, color='black', linestyle='--', label='Desired speed')

    # Add maximum episode steps line
    if metric == 'eval/mean_ep_steps':
        plt.axhline(y=1000, color='black', linestyle='--', label='Max episode steps')

    # Y-axis scaling
    if metric == 'train_critic/loss':
        plt.yscale('log')

    # X-axis label
    plt.xlabel('Environment steps (Millions)')

    # Y-axis label  
    plt.ylabel(get_y_label(metric))

    # Grid and legend
    plt.grid()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    lgd = ax.legend(loc='center right', bbox_to_anchor=(1.33, 0.5), fancybox=True, shadow=True)
    plt.savefig(metric.replace('/', '_') + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')