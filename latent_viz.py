import os
import json
import utils
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

import warnings
warnings.filterwarnings(action='ignore')

WEATHER_PRESETS =  {
                    # Training weather presets
                    0: 'ClearNoon',
                    1: 'ClearSunset', 
                    2: 'CloudyNoon', 
                    3: 'CloudySunset', 
                    4: 'WetNoon', 
                    5: 'WetSunset', 
                    6: 'MidRainSunset',
                    # Novel weather presets
                    7: 'MidRainyNoon',
                    8: 'WetCloudySunset',
                    9: 'HardRainNoon'
                    }

# Greenish, blueish colors for indexes 0-6, redish, yellowish colors for indexes 7-9
weather_colors = ['#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#008000', '#008080', '#000080', '#ff0000', '#ffff00', '#800000']

def get_experiment_title(name):
    title = None
    if name == 'pixel_sac':
        title = 'Experiment: Pixel SAC'
    elif name == 'identity':
        title = 'Experiment: CURL-SAC with identity augmentation'
    elif name == 'random_crop':
        title = 'Experiment: CURL-SAC with random crop augmentation'
    elif name == 'color_jiggle':
        title = 'Experiment: CURL-SAC with color jiggle augmentation'
    elif name == 'noisy_cover':
        title = 'Experiment: CURL-SAC with noisy cover augmentation'
    return title

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir_path', default='', type=str)
    parser.add_argument('--model_step', default=1_000_000, type=int)
    args = parser.parse_args()
    return args

def plot_latent_tsne(file_path, file_name):
    
    # Load latent_value_dict dictionary from file
    with open(os.path.join(file_path, file_name), 'r') as f:
        latent_value_dict = json.load(f)

    # Get the latent representations and q values in the right format
    representations = []
    q_values = []
    weather_presets = []
    for step in latent_value_dict.keys():
        representations.append(np.array(latent_value_dict[step]['latent_representation']))
        q_values.append(latent_value_dict[step]['q_value'])
        weather_presets.append(latent_value_dict[step]['weather_preset_idx'])
    assert len(representations) == len(q_values)
    representations = np.array(representations, dtype=np.float32)
    q_values = np.array(q_values, dtype=np.float32)
    weather_presets = np.array(weather_presets, dtype=np.uint8)

    # Print some statistics
    print('-'*50)
    print(f'Mean Q-value: {np.mean(q_values)}')
    print(f'Median Q-value: {np.median(q_values)}')
    print(f'Max Q-value: {np.max(q_values)}')
    print(f'Min Q-value: {np.min(q_values)}')
    print(f'Standard deviation Q-value: {np.std(q_values)}')
    nb_std = 1
    max_tick = int(np.mean(q_values) + nb_std*np.std(q_values))
    min_tick = int(np.mean(q_values) - nb_std*np.std(q_values))
    print('-'*50)

    # Perform t-SNE on the latent representations
    tsne = TSNE(n_components=2)
    print('Performing t-SNE on the latent representations...')
    vec_2d = tsne.fit_transform(representations)
    print('Done.')

    # Figure settings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    marker_size = 5

    # Visualize latent representations in function of Q-values
    scatter = ax1.scatter(vec_2d[:,0], vec_2d[:,1], c=q_values, cmap='viridis', s=marker_size, vmin=min_tick, vmax=max_tick)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax1, location='left', extend='both')
    cbar.set_label('Q Value')

    # Visualize latent representations in function of weather presets
    cmap = plt.cm.colors.ListedColormap(weather_colors)
    scatter = ax2.scatter(vec_2d[:,0], vec_2d[:,1], c=weather_presets, cmap=cmap, s=marker_size, vmin=0, vmax=len(WEATHER_PRESETS))
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_ticks(np.arange(len(WEATHER_PRESETS)) + 0.5)
    cbar.set_ticklabels(WEATHER_PRESETS.values())
    cbar.set_label('Weather Preset')

    # Save the figure
    exp_name = file_name.split('.')[0]
    title = get_experiment_title(exp_name)
    fig.suptitle(f'{title}')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'{exp_name}.png'))

def main():

    # Parse arguments
    args = parse_args()
    with open(os.path.join(args.experiment_dir_path, 'args.json'), 'r') as f:
        args.__dict__.update(json.load(f))

    # Set a fixed random seed for reproducibility across weather presets.
    args.seed = 0
    utils.set_seed_everywhere(args.seed)

    # Iterate over every .json file in the latent_data directory
    for file_name in os.listdir('./latent_data'):
        if file_name.endswith('.json') and str(args.augmentation) in file_name:
            file_path = './latent_data'
            print(f'Plotting latent t-SNE for {file_name}')
            plot_latent_tsne(file_path, file_name)

if __name__ == '__main__':
    main()