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

def run_episode(env, agent, augmentor, step, experiment_dir_path, weather_name):
        
        # Initializations
        path = os.path.join(experiment_dir_path, 'latent_data')
        if not os.path.exists(path):
            os.mkdir(path)
        video = VideoRecorder(path, env.fps)

        latent_value_dict = {}

        # Run evaluation loop
        obs = env.reset()
        video.init(enabled=True)
        done = False
        step = 0
        while not done:

            # Perform anchor augmentation
            obs = augmentor.anchor_augmentation(obs)

            # Get the latent representation
            latent_representation = agent.actor.encoder(obs)

            # Sample action from agent
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

            # Get the values
            q1 = agent.critic.Q1(latent_representation, action)
            q2 = agent.critic.Q2(latent_representation, action)

            # Save the latent representation and the values
            latent_representation = latent_representation.detach().cpu().numpy()
            q1 = q1.detach().cpu().numpy()
            q2 = q2.detach().cpu().numpy()
            q = float(np.mean([q1, q2], axis=0))
            latent_value_dict[step] = {'latent_representation': latent_representation, 'q_value': q}

            # Take step in environment
            obs, reward, done, _ = env.step(action)
            step += 1

            # Administration and logging
            video.record(env)
        
        # Save the video
        video.save(f'{weather_name}.mp4')

        # Save latent_value_dict dictionary to file
        with open(os.path.join(path, f'{weather_name}.json'), 'w') as f:
            json.dump(latent_value_dict, f)

def main():

    # Parse arguments
    args = parse_args()
    with open(os.path.join(args.experiment_dir_path, 'args.json'), 'r') as f:
        args.__dict__.update(json.load(f))

    # Set a fixed random seed for reproducibility across weather presets.
    args.seed = 0

    # Set shorter episode time
    args.seconds_per_episode = 20

    # Random seed
    utils.set_seed_everywhere(args.seed)

    # Anchor/target data augmentor
    camera_image_shape = (args.camera_image_height, args.camera_image_width)
    augmentor = make_augmentor(args.augmentation, camera_image_shape)

    # Perform the same episode for each weather preset
    WEATHER_PRESETS =  {'ClearNoon': carla.WeatherParameters.ClearNoon,
                        'ClearSunset': carla.WeatherParameters.ClearSunset, 
                        'CloudyNoon': carla.WeatherParameters.CloudyNoon, 
                        'CloudySunset': carla.WeatherParameters.CloudySunset, 
                        'WetNoon': carla.WeatherParameters.WetNoon, 
                        'WetSunset': carla.WeatherParameters.WetSunset, 
                        'MidRainSunset': carla.WeatherParameters.MidRainSunset,
                        'MidRainyNoon': carla.WeatherParameters.MidRainyNoon}
    
    # Thanks to the fixed random seed, the same episode will be performed for each 
    # weather preset if the CARLA server process is killed in between
    for weather_name in WEATHER_PRESETS:

        # Launch the CARLA server and load the model
        env = make_env(args)
        action_shape = env.action_space.shape
        pre_aug_obs_shape = env.observation_space.shape
        obs_shape = (3*args.frame_stack, args.augmented_image_height, args.augmented_image_width)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = make_agent(obs_shape, action_shape, args, device, augmentor)
        model_dir_path = os.path.join(args.experiment_dir_path, 'model')
        agent.load(model_dir_path, str(args.augmentation), str(args.model_step))

        # Set the weather preset
        env.weather_presets = [WEATHER_PRESETS[weather_name],]

        # Run the episode
        run_episode(env, agent, augmentor, args.model_step, args.experiment_dir_path, weather_name)

        # Deactivate the environment (kills the CARLA server)
        env.deactivate()

if __name__ == "__main__":
    main()