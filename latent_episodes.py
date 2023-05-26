import os
import argparse
import torch
import numpy as np
import json
import carla

import utils
from augmentations import make_augmentor
import carla_env
from video import VideoRecorder
from train import make_agent
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir_path', default='', type=str)
    parser.add_argument('--model_step', default=1_000_000, type=int)
    args = parser.parse_args()
    return args

def make_env(args):

    # Initialize the CARLA environment
    env = carla_env.CarlaEnv(args.carla_town, args.max_npc_vehicles, 
                   args.desired_speed, args.max_stall_time, args.stall_speed, args.seconds_per_episode,
                   args.fps, 2000, 8000, args.env_verbose, args.camera_image_height, args.camera_image_width, 
                   args.fov, args.cam_x, args.cam_y, args.cam_z, args.cam_pitch,
                   args.lambda_r1, args.lambda_r2, args.lambda_r3, args.lambda_r4, args.lambda_r5)
    
    # Set the random seed and reset
    env.seed(args.seed)
    env.reset()

    # Wrap CarlaEnv in FrameStack class to stack several consecutive frames together
    env = utils.FrameStack(env, k=args.frame_stack)

    return env

class CustomReplayBuffer(Dataset):
    
    def __init__(self, obs_shape, action_shape, capacity):

        # Initialize the replay buffer
        self.capacity = capacity
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.uint8)
        self.speeds = np.empty((capacity, 1), dtype=np.float32)
        self.weather_preset_idxs = np.empty((capacity, 1), dtype=np.uint8)

        # Initialize the buffer administration
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, done, speed, weather_preset_idx):
            
            # Add data to the replay buffer
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.actions[self.idx], action)
            np.copyto(self.rewards[self.idx], reward)
            np.copyto(self.dones[self.idx], done)
            np.copyto(self.speeds[self.idx], speed)
            np.copyto(self.weather_preset_idxs[self.idx], weather_preset_idx)
    
            # Update administration
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0

    def save(self, path):
        # Save the replay buffer
        np.savez_compressed(path, 
                            obses=self.obses, 
                            actions=self.actions, 
                            rewards=self.rewards, 
                            dones=self.dones, 
                            speeds=self.speeds, 
                            weather_preset_idxs=self.weather_preset_idxs)

    def load(self, path):
        # Load the replay buffer
        with np.load(path) as f:
            self.obses = f['obses']
            self.actions = f['actions']
            self.rewards = f['rewards']
            self.dones = f['dones']
            self.speeds = f['speeds']
            self.weather_preset_idxs = f['weather_preset_idxs']

def run_episodes(env, agent, augmentor, nb_steps=1000):
        
        # Initializations
        path = './latent_data'
        if not os.path.exists(path):
            os.mkdir(path)
        video = VideoRecorder(path, env.fps)
        done = True
        episode = 0
        replay_buffer = CustomReplayBuffer(env.observation_space.shape, env.action_space.shape, nb_steps)

        for step in range(nb_steps):

            if step % 1000 == 0:
                print(f'Step {step}/{nb_steps}')

            if done:
                if episode > 0:
                    video.save(f'episode_{episode}_{int(cumul_reward)}.mp4')
                obs = env.reset()
                weather_preset_idx = env.weather_preset_idx
                video.init(enabled=True)
                done = False
                episode += 1
                episode_step = 0
                cumul_reward = 0

            # Perform anchor augmentation
            obs = augmentor.anchor_augmentation(obs)

            # Sample action from agent
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

            # Take step in environment
            obs, reward, done, _ = env.step(action)
            speed = env.abs_kmh
            cumul_reward += reward
            episode_step += 1

            # Save data to replay buffer
            replay_buffer.add(obs, action, reward, done, speed, weather_preset_idx)

            # Administration and logging
            video.record(env)

            if step + 2 == nb_steps: 
                done = True

        # Save the replay buffer
        replay_buffer.save(os.path.join(path, 'replay_buffer.npz'))

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

    # Launch the CARLA server and load the model
    env = make_env(args)
    action_shape = env.action_space.shape
    pre_aug_obs_shape = env.observation_space.shape
    obs_shape = (3*args.frame_stack, args.augmented_image_height, args.augmented_image_width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = make_agent(obs_shape, action_shape, args, device, augmentor)
    model_dir_path = os.path.join(args.experiment_dir_path, 'model')
    agent.load(model_dir_path, str(args.augmentation), str(args.model_step))

    # Run the episode
    run_episodes(env, agent, augmentor, nb_steps=10000)

    # Deactivate the environment (kills the CARLA server)
    env.deactivate()

if __name__ == "__main__":
    main()