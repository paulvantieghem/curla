import os
import argparse
import torch
import numpy as np
import json
import carla
from tqdm import tqdm

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

    # Initialize dictionary to store latent representations and values
    latent_value_dict = {}

    # Loop over replay buffer and save latent representation and values
    for i in tqdm(range(replay_buffer.obses.shape[0])):

        # Get data from replay buffer
        obs = replay_buffer.obses[i]
        action = replay_buffer.actions[i]
        reward = float(replay_buffer.rewards[i])
        done = int(replay_buffer.dones[i])
        speed = int(replay_buffer.speeds[i])
        weather_preset_idx = int(replay_buffer.weather_preset_idxs[i])

        # Get the weather preset name
        weather_preset_idx

        # Perform anchor augmentation
        obs = augmentor.anchor_augmentation(obs)

        # Get the latent representation
        with torch.no_grad():
            gpu_obs = torch.FloatTensor(obs).to('cuda')
            gpu_obs = gpu_obs.unsqueeze(0)
            latent_representation = agent.actor.encoder(gpu_obs)

        # Get the models own action (OPTIONAL)
        # with utils.eval_mode(agent):
        #     action = agent.sample_action(obs)
        
        # Get the values
        with torch.no_grad():
            gpu_action = torch.FloatTensor(action).to('cuda')
            gpu_action = gpu_action.unsqueeze(0)
            q1 = agent.critic.Q1(latent_representation, gpu_action)
            q2 = agent.critic.Q2(latent_representation, gpu_action)

        # Save the latent representation and the values
        latent_representation = list(latent_representation.detach().cpu().numpy().squeeze())
        latent_representation = [float(x) for x in latent_representation]
        q1 = q1.detach().cpu().numpy()
        q2 = q2.detach().cpu().numpy()
        q = float(np.min([q1, q2], axis=0))
        latent_value_dict[i] = {'latent_representation': latent_representation, 
                                'q_value': q,
                                'reward': reward,
                                'speed': speed,
                                'weather_preset_idx': weather_preset_idx}

    # Save latent_value_dict dictionary to file
    augmentation_name = str(args.augmentation)
    with open(os.path.join('latent_data', f'{augmentation_name}.json'), 'w') as f:
        json.dump(latent_value_dict, f)


if __name__ == "__main__":
    main()