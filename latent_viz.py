import os
import json
import utils
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def get_closest_obs_diff_weather(idx, representations, replay_buffer):

    # Get the closest observation in latent space to the idx-th observation with a different weather preset
    idxes = []
    preset1 = WEATHER_PRESETS[replay_buffer.weather_preset_idxs[idx][0]]
    for i in range(len(replay_buffer.weather_preset_idxs)):
        if replay_buffer.weather_preset_idxs[i] != replay_buffer.weather_preset_idxs[idx]:
            preset2 = WEATHER_PRESETS[replay_buffer.weather_preset_idxs[i][0]]
            if 'Noon' in preset1 and 'Sunset' in preset2:
                idxes.append(i)
            elif 'Sunset' in preset1 and 'Noon' in preset2:
                idxes.append(i)
    idxes = np.array(idxes)
    idxes = idxes[np.argsort(np.linalg.norm(representations[idxes] - representations[idx], axis=1))]
    idx1 = idxes[0]
    distance1 = np.linalg.norm(representations[idx1] - representations[idx])

    return [idx, idx1], distance1

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
    elif name == 'identity_detached':
        title = 'Experiment: CURL-SAC with identity augmentation and detached encoder'
    return title

def get_image_from_obs(obs):
    obs = obs[0:3, :, :]
    obs = np.transpose(obs, (1, 2, 0))
    return obs

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

    # ###################################################
    # ### Specific observations in latent space plots ###
    # ###################################################
    idxes1, distance1 = get_closest_obs_diff_weather(7108, representations, replay_buffer)
    idxes2, distance2 = get_closest_obs_diff_weather(2565, representations, replay_buffer)
    print(f'Distance between observations1: {distance1}')
    print(f'Distance between observations2: {distance2}')

    # Process the corresponding observations
    observations1 = [replay_buffer.obses[idx] for idx in idxes1]
    for i in range(len(observations1)):
        observations1[i] = get_image_from_obs(observations1[i])
    observations2 = [replay_buffer.obses[idx] for idx in idxes2]
    for i in range(len(observations2)):
        observations2[i] = get_image_from_obs(observations2[i])


    # Plot the corresponding images
    plt.figure(figsize=(8, 6), dpi=200)
    for i in range(len(observations1)):
        plt.subplot(2, 2, i+1)
        plt.imshow(observations1[i])
        if i == 0:
            plt.title(f'Weather: {WEATHER_PRESETS[replay_buffer.weather_preset_idxs[idxes1[i]][0]]} (O)', color='red')
        else:
            plt.title(f'Weather: {WEATHER_PRESETS[replay_buffer.weather_preset_idxs[idxes1[i]][0]]} (X)', color='red')
        plt.axis('off')
    for i in range(len(observations2)):
        plt.subplot(2, 2, i+3)
        plt.imshow(observations2[i])
        if i == 0:
            plt.title(f'Weather: {WEATHER_PRESETS[replay_buffer.weather_preset_idxs[idxes2[i]][0]]} (O)', color='blue')
        else:
            plt.title(f'Weather: {WEATHER_PRESETS[replay_buffer.weather_preset_idxs[idxes2[i]][0]]} (X)', color='blue')
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

    # Draw a red cross at points in idxes1 and a red line to connect them
    ax1.scatter(tsne_representations[idxes1[0],0], tsne_representations[idxes1[0],1], s=60, facecolors='none', edgecolors='red', linewidths=2, zorder=10)
    ax1.scatter(tsne_representations[idxes1[1],0], tsne_representations[idxes1[1],1], s=60, marker='x', c='red', linewidths=2, zorder=10)
    ax1.plot([tsne_representations[idxes1[0],0], tsne_representations[idxes1[1],0]], [tsne_representations[idxes1[0],1], tsne_representations[idxes1[1],1]], color='red', linewidth=2, zorder=10)
    # Draw a blue cross at points in idxes2 and a blue line to connect them
    ax1.scatter(tsne_representations[idxes2[0],0], tsne_representations[idxes2[0],1], s=60, facecolors='none', edgecolors='blue', linewidths=2, zorder=10)
    ax1.scatter(tsne_representations[idxes2[1],0], tsne_representations[idxes2[1],1], s=60, marker='x', c='blue', linewidths=2, zorder=10)
    ax1.plot([tsne_representations[idxes2[0],0], tsne_representations[idxes2[1],0]], [tsne_representations[idxes2[0],1], tsne_representations[idxes2[1],1]], color='blue', linewidth=2, zorder=10)
    
    
    # Visualize latent representations in function of weather presets
    scatter = ax2.scatter(tsne_representations[:,0], tsne_representations[:,1], 
                          c=replay_buffer.weather_preset_idxs, cmap='tab10', s=marker_size, vmin=0, vmax=len(WEATHER_PRESETS))
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_ticks(np.arange(len(WEATHER_PRESETS)) + 0.5)
    cbar.set_ticklabels(WEATHER_PRESETS.values())
    cbar.set_label('Weather Preset')

    # Draw a red cross at points in idxes1 and a red line to connect them
    # Draw a red cross at points in idxes1 and a red line to connect them
    ax2.scatter(tsne_representations[idxes1[0],0], tsne_representations[idxes1[0],1], s=60, facecolors='none', edgecolors='red', linewidths=2, zorder=10)
    ax2.scatter(tsne_representations[idxes1[1],0], tsne_representations[idxes1[1],1], s=60, marker='x', c='red', linewidths=2, zorder=10)
    ax2.plot([tsne_representations[idxes1[0],0], tsne_representations[idxes1[1],0]], [tsne_representations[idxes1[0],1], tsne_representations[idxes1[1],1]], color='red', linewidth=2, zorder=10)
    # Draw a blue cross at points in idxes2 and a blue line to connect them
    ax2.scatter(tsne_representations[idxes2[0],0], tsne_representations[idxes2[0],1], s=60, facecolors='none', edgecolors='blue', linewidths=2, zorder=10)
    ax2.scatter(tsne_representations[idxes2[1],0], tsne_representations[idxes2[1],1], s=60, marker='x', c='blue', linewidths=2, zorder=10)
    ax2.plot([tsne_representations[idxes2[0],0], tsne_representations[idxes2[1],0]], [tsne_representations[idxes2[0],1], tsne_representations[idxes2[1],1]], color='blue', linewidth=2, zorder=10)

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