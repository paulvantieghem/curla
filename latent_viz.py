import os
import json
import utils
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

def find_nearest(array, value):
    idx = (np.linalg.norm(array - value, axis=1)).argmin()
    return idx

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

    # Get experiment name
    exp_name = file_name.split('.')[0]

    ###################################################
    ### Specific observations in latent space plots ###
    ###################################################
    if exp_name == 'pixel_sac':
        point1 = np.array([])
        point2 = np.array([])
        point3 = np.array([])
    elif exp_name == 'identity':
        point1 = np.array([])
        point2 = np.array([])
        point3 = np.array([])
    elif exp_name == 'random_crop':
        point1 = np.array([])
        point2 = np.array([])
        point3 = np.array([])
    elif exp_name == 'color_jiggle':
        point1 = np.array([-57.124, 51.5779])
        point2 = np.array([-59.461, 51.8416])
        point3 = np.array([-59.007, 52.0403])
    elif exp_name == 'noisy_cover':
        point1 = np.array([])
        point2 = np.array([])
        point3 = np.array([])
    
    # Get average point to plot as a large red circle
    avg_point = np.mean(np.stack([point1, point2, point3]), axis=0)

    # Get indices of closest points found in t-SNE figure
    idx1 = find_nearest(tsne_representations, point1)
    idx2 = find_nearest(tsne_representations, point2)
    idx3 = find_nearest(tsne_representations, point3)
    idxes = [idx1, idx2, idx3]

    # Process the corresponding observations
    observations = [replay_buffer.obses[idx1],
                replay_buffer.obses[idx2],
                replay_buffer.obses[idx3]]
    for i in range(len(observations)):
        obs = observations[i]
        obs = obs[0:3, :, :]
        obs = np.transpose(obs, (1, 2, 0))
        observations[i] = obs

    # Plot the corresponding images
    plt.figure(figsize=(12, 3), dpi=200)
    for i in range(len(observations)):
        plt.subplot(1, 3, i+1)
        plt.imshow(observations[i])
        plt.title(f'Weather: {WEATHER_PRESETS[replay_buffer.weather_preset_idxs[idxes[i]][0]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'tsne_{exp_name}_images.png'))

    ###########################################################
    ### t-SNE plot of latents by Q-value and weather preset ###
    ###########################################################

    # Figure settings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    marker_size = 5

    # Visualize latent representations in function of Q-values
    scatter = ax1.scatter(tsne_representations[:,0], tsne_representations[:,1], 
                          c=q_values, cmap='viridis', s=marker_size, vmin=min_tick, vmax=max_tick)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax1, location='left', extend='both')
    cbar.set_label('Q Value')

    # Add avg_point to the plot as a large red circle
    ax1.scatter(avg_point[0], avg_point[1], s=60, facecolors='none', edgecolors='red', linewidths=2, zorder=10)

    # Visualize latent representations in function of weather presets
    scatter = ax2.scatter(tsne_representations[:,0], tsne_representations[:,1], 
                          c=replay_buffer.weather_preset_idxs, cmap='tab10', s=marker_size, vmin=0, vmax=len(WEATHER_PRESETS))
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_ticks(np.arange(len(WEATHER_PRESETS)) + 0.5)
    cbar.set_ticklabels(WEATHER_PRESETS.values())
    cbar.set_label('Weather Preset')

    # Add avg_point to the plot as a large red circle
    ax2.scatter(avg_point[0], avg_point[1], s=60, facecolors='none', edgecolors='red', linewidths=2, zorder=10)

    # Save the figure
    title = get_experiment_title(exp_name)
    fig.suptitle(f'{title}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(file_path, f'tsne_{exp_name}_values_weather.png'))

    #################################################
    ### t-SNE plot of latents by reward and speed ###
    #################################################

    # # Figure settings
    # fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(7, 3), dpi=200)
    # marker_size = 5

    # # Visualize latent representations in function of replay_buffer.rewards
    # scatter = ax3.scatter(tsne_representations[:,0], tsne_representations[:,1], 
    #                       c=replay_buffer.rewards, cmap='viridis', s=marker_size, vmin=np.mean(replay_buffer.rewards)-np.std(replay_buffer.rewards), vmax=np.mean(replay_buffer.rewards)+np.std(replay_buffer.rewards))
    # ax3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    # cbar = plt.colorbar(scatter, ax=ax3)
    # cbar.set_label('Reward')

    # # Visualize latent representations in function of replay_buffer.speeds
    # scatter = ax4.scatter(tsne_representations[:,0], tsne_representations[:,1], 
    #                       c=replay_buffer.speeds, cmap='viridis', s=marker_size)
    # ax4.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    # cbar = plt.colorbar(scatter, ax=ax4)
    # cbar.set_label('Speed')

    # # Save the figure
    # fig.suptitle(f'{title}')
    # plt.tight_layout()
    # plt.savefig(os.path.join(file_path, f'tsne_{exp_name}_rewards_speeds.png'))


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