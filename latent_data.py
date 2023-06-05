import os
import argparse
import torch
import numpy as np
import json
import carla
from tqdm import tqdm
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings(action='ignore')

import utils
from augmentations import make_augmentor
import carla_env
from video import VideoRecorder
from train import make_agent
from latent_episodes import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir_path', default='', type=str)
    parser.add_argument('--model_step', default=1_000_000, type=int)
    args = parser.parse_args()
    return args

def main():

    # Parse arguments
    args = parse_args()
    with open(os.path.join(args.experiment_dir_path, 'args.json'), 'r') as f:
        args.__dict__.update(json.load(f))

    # Set a fixed random seed for reproducibility across weather presets.
    args.seed = 0

    if not hasattr(args, 'pixel_sac'):
        args.pixel_sac = False

    # Random seed
    utils.set_seed_everywhere(args.seed)

    # Anchor/target data augmentor
    camera_image_shape = (args.camera_image_height, args.camera_image_width)
    augmentor = make_augmentor(args.augmentation, camera_image_shape)
    
    # Load the CURL model
    action_shape = (2,)
    obs_shape = (3*args.frame_stack, args.augmented_image_height, args.augmented_image_width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = make_agent(obs_shape, action_shape, args, device, augmentor)
    model_dir_path = os.path.join(args.experiment_dir_path, 'model')
    agent.load(model_dir_path, str(args.augmentation), str(args.model_step))

    # Load replay buffer
    replay_buffer = CustomReplayBuffer(obs_shape=obs_shape, action_shape=action_shape, capacity=10_000)
    replay_buffer.load('./latent_data/replay_buffer.npz')

    # Initialize arrays
    representations = []
    q_values = []

    # Loop over replay buffer and save latent representation and values
    for i in tqdm(range(replay_buffer.obses.shape[0])):

        # Get data from replay buffer
        obs = replay_buffer.obses[i]
        # action = replay_buffer.actions[i]
        reward = float(replay_buffer.rewards[i])
        done = int(replay_buffer.dones[i])
        speed = int(replay_buffer.speeds[i])
        weather_preset_idx = int(replay_buffer.weather_preset_idxs[i])

        # Get the weather preset name
        weather_preset_idx

        # Perform anchor augmentation
        obs = augmentor.evaluation_augmentation(obs)

        # Get the latent representation
        with torch.no_grad():
            gpu_obs = torch.FloatTensor(obs).to('cuda')
            gpu_obs = gpu_obs.unsqueeze(0)
            latent_representation = agent.actor.encoder(gpu_obs)

        # Get the model's own action for the calculation of the Q-values
        with utils.eval_mode(agent):
            action = agent.sample_action(obs)
        
        # Get the values
        with torch.no_grad():
            gpu_action = torch.FloatTensor(action).to('cuda')
            gpu_action = gpu_action.unsqueeze(0)
            q1, q2 = agent.critic(gpu_obs, gpu_action)

        # Save the latent representation and the values
        latent_representation = list(latent_representation.detach().cpu().numpy().squeeze())
        latent_representation = [float(x) for x in latent_representation]
        q1 = q1.detach().cpu().numpy()
        q2 = q2.detach().cpu().numpy()
        q = float(np.min([q1, q2], axis=0))
        
        # Add data to arrays
        representations.append(latent_representation)
        q_values.append(q)

    # Convert arrays to numpy arrays
    representations = np.array(representations, dtype=np.float32)
    q_values = np.array(q_values, dtype=np.float32)

    # # Calculate the cosine similarity between the latent representations
    # from sklearn.metrics.pairwise import cosine_similarity
    # S = cosine_similarity(representations)
    # D = 1. - S

    # Perform t-SNE on the latent representations
    tsne = TSNE(n_components=2)
    print('Performing t-SNE on the latent representations...')
    tsne_representations = tsne.fit_transform(representations)
    # tsne_representations = tsne.fit_transform(D)
    print('Done.')

    # # Use PCA to reduce the dimensionality of the latent representations
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # print('Performing PCA on the latent representations...')
    # pca_representations = pca.fit_transform(representations)
    # print('Done.')


    # Save latent_value_dict dictionary to file
    exp_name = args.experiment_dir_path.split('-')[-1]
    np.savez_compressed(f'./latent_data/{exp_name}.npz',
                        representations=representations,
                        q_values=q_values,
                        tsne_representations=tsne_representations)
    
if __name__ == "__main__":
    main()