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

WEATHER_PRESETS =  {0: 'ClearNoon',
                    1: 'ClearSunset', 
                    2: 'CloudyNoon', 
                    3: 'CloudySunset', 
                    4: 'WetNoon', 
                    5: 'WetSunset', 
                    6: 'MidRainSunset'}

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

    # Perform t-SNE on the latent representations
    tsne = TSNE(n_components=2)
    vec_2d = tsne.fit_transform(representations)

    # Visualize latent representations in function of Q-values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    scatter = ax1.scatter(vec_2d[:,0], vec_2d[:,1], c=q_values, cmap='viridis', s=10)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax1, location='left')
    cbar.ax.set_yticklabels(np.linspace(int(np.min(q_values)), int(np.max(q_values)), 10, endpoint=False).astype(np.int32))
    cbar.set_label('Q Value')

    # Visualize latent representations in function of weather presets
    scatter = ax2.scatter(vec_2d[:,0], vec_2d[:,1], c=weather_presets, cmap='tab10', s=10)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.ax.set_yticklabels(WEATHER_PRESETS.values())
    cbar.set_label('Weather Preset')

    # Save the figure
    title = file_name.split('.')[0]
    fig.suptitle(f'Augmentation: {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'{title}.png'))

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