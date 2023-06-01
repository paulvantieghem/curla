import os
import json
import utils
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from latent_episodes import CustomReplayBuffer

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
                    7: 'MidRainyNoon*',
                    8: 'WetCloudySunset*',
                    9: 'HardRainNoon*'}

SELECTED_OBSERVATIONS = {'scenario_1': [5035, 7109, 0],
                         'scenario_2': [0, 0, 0],
                         'scenario_3': [0, 0, 0]}

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
    
    # Load replay buffer
    replay_buffer = CustomReplayBuffer()
    replay_buffer.load('./latent_data/replay_buffer.npz')

    # Load latent values
    with np.load(os.path.join(file_path, file_name)) as f:
        q_values = f['q_values']
        representations = f['representations']
        tsne_representations = f['tsne_representations']

    # Visualize selected observations
    for scenario in SELECTED_OBSERVATIONS:
        fig, axs = plt.subplots(1, 3, figsize=(8, 4))
        for i in range(len(axs)):
            obs = replay_buffer.obses[SELECTED_OBSERVATIONS[scenario][i], 0:3, :, :]
            obs = obs.transpose(1, 2, 0)
            axs[i].imshow(obs)
            axs[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            axs[i].set_title(f'Observation {SELECTED_OBSERVATIONS[scenario][i]}')
        plt.tight_layout()
        plt.show()


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

    # Figure settings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    marker_size = 5

    # Visualize latent representations in function of Q-values
    scatter = ax1.scatter(tsne_representations[:,0], tsne_representations[:,1], 
                          c=q_values, cmap='viridis', s=marker_size, vmin=min_tick, vmax=max_tick)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax1, location='left', extend='both')
    cbar.set_label('Q Value')

    # Visualize selected observations
    marker_size_factor = 10
    scatter_1 = ax1.scatter(tsne_representations[SELECTED_OBSERVATIONS['scenario_1'],0], 
                            tsne_representations[SELECTED_OBSERVATIONS['scenario_1'],1], 
                            c='red', marker='o', s=marker_size*marker_size_factor, facecolors='none', edgecolors='red')
    for i, txt in enumerate(SELECTED_OBSERVATIONS['scenario_1']):
        ax1.annotate(str(i), (tsne_representations[SELECTED_OBSERVATIONS['scenario_1'][i],0], 
                           tsne_representations[SELECTED_OBSERVATIONS['scenario_1'][i],1]))
        
    # scatter_2 = ax1.scatter(tsne_representations[SELECTED_OBSERVATIONS['scenario_2'],0], 
    #                         tsne_representations[SELECTED_OBSERVATIONS['scenario_2'],1], 
    #                         c='red', marker='s', s=marker_size*marker_size_factor, facecolors='none', edgecolors='red')
    # for i, txt in enumerate(SELECTED_OBSERVATIONS['scenario_2']):
    #     ax1.annotate(str(i), (tsne_representations[SELECTED_OBSERVATIONS['scenario_2'][i],0], 
    #                        tsne_representations[SELECTED_OBSERVATIONS['scenario_2'][i],1]))
        
    # scatter_3 = ax1.scatter(tsne_representations[SELECTED_OBSERVATIONS['scenario_3'],0], 
    #                         tsne_representations[SELECTED_OBSERVATIONS['scenario_3'],1], 
    #                         c='red', marker='^', s=marker_size*marker_size_factor, facecolors='none', edgecolors='red')
    # for i, txt in enumerate(SELECTED_OBSERVATIONS['scenario_3']):
    #     ax1.annotate(str(i), (tsne_representations[SELECTED_OBSERVATIONS['scenario_3'][i],0], 
    #                        tsne_representations[SELECTED_OBSERVATIONS['scenario_3'][i],1]))

    # Visualize latent representations in function of weather presets
    scatter = ax2.scatter(tsne_representations[:,0], tsne_representations[:,1], 
                          c=replay_buffer.weather_preset_idxs, cmap='tab10', s=marker_size, vmin=0, vmax=len(WEATHER_PRESETS))
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
    plt.savefig(os.path.join(file_path, f'tsne_{exp_name}_values_weather.png'))

    # Figure settings
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    marker_size = 5

    # Visualize latent representations in function of replay_buffer.rewards
    scatter = ax3.scatter(tsne_representations[:,0], tsne_representations[:,1], 
                          c=replay_buffer.rewards, cmap='viridis', s=marker_size, vmin=np.mean(replay_buffer.rewards)-np.std(replay_buffer.rewards), vmax=np.mean(replay_buffer.rewards)+np.std(replay_buffer.rewards))
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Reward')

    # Visualize latent representations in function of replay_buffer.speeds
    scatter = ax4.scatter(tsne_representations[:,0], tsne_representations[:,1], 
                          c=replay_buffer.speeds, cmap='viridis', s=marker_size)
    ax4.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Speed')

    # Save the figure
    fig.suptitle(f'{title}')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'tsne_{exp_name}_replay_buffer.rewards_replay_buffer.speeds.png'))


def main():

    # Parse arguments
    args = parse_args()
    with open(os.path.join(args.experiment_dir_path, 'args.json'), 'r') as f:
        args.__dict__.update(json.load(f))

    # Set a fixed random seed for reproducibility across weather presets.
    args.seed = 0
    utils.set_seed_everywhere(args.seed)

    if not hasattr(args, 'pixel_sac'):
        args.pixel_sac = False

    # Iterate over every .json file in the latent_data directory
    for file_name in os.listdir('./latent_data'):
        exp_name = 'pixel_sac' if args.pixel_sac else str(args.augmentation)
        if file_name.endswith('.npz') and exp_name in file_name:
            file_path = './latent_data'
            print(f'Plotting latent t-SNE for {file_name}')
            plot_latent_tsne(file_path, file_name)

if __name__ == '__main__':
    main()