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

USE_NOVEL_PRESETS = True

if USE_NOVEL_PRESETS:
    # Evaluation weather presets
    WEATHER_PRESETS =  {'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
                        'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
                        'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
                        'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
                        'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
                        'HardRainNoon': carla.WeatherParameters.HardRainNoon,
                        'HardRainSunset': carla.WeatherParameters.HardRainSunset}

else:
    # Training weather presets
    WEATHER_PRESETS = {'ClearNoon': carla.WeatherParameters.ClearNoon,
                        'ClearSunset': carla.WeatherParameters.ClearSunset,
                        'CloudyNoon': carla.WeatherParameters.CloudyNoon,
                        'CloudySunset': carla.WeatherParameters.CloudySunset,
                        'WetNoon': carla.WeatherParameters.WetNoon,
                        'WetSunset': carla.WeatherParameters.WetSunset,
                        'MidRainSunset': carla.WeatherParameters.MidRainSunset}
                    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir_path', default='', type=str)
    parser.add_argument('--model_step', default=1_000_000, type=int)
    parser.add_argument('--env_verbose', default=False, action='store_true')
    args = parser.parse_args()
    return args


def run_eval_loop(env, agent, augmentor, step, experiment_dir_path, num_episodes=10, record_video=False):
        
        # Initializations
        exp_name = os.path.basename(experiment_dir_path)
        exp_name = exp_name.split('-')[-1]
        print(f'Running evaluation loop for experiment {exp_name}')
        ep_rewards = []
        ep_steps = []
        path = os.path.join(experiment_dir_path, 'eval_videos')
        if record_video:
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))
        video = VideoRecorder(path, env.fps)

        # Run evaluation loop
        for i in range(num_episodes):
            obs = env.reset()
            chosen_preset = list(WEATHER_PRESETS.keys())[env.weather_preset_idx]
            video.init(enabled=record_video)
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:

                # Perform anchor augmentation
                obs = augmentor.evaluation_augmentation(obs)

                # Sample action from agent
                with utils.eval_mode(agent):
                    action = agent.sample_action(obs)

                # Take step in environment
                obs, reward, done, info = env.step(action)

                # Administration and logging
                video.record(env)
                episode_reward += reward
                episode_step += 1
                    
            video.save(f'{step}_{i+1}_r{int(episode_reward)}.mp4')
            ep_steps.append(episode_step)
            ep_rewards.append(episode_reward)
            print('Episode %d/%d | Weather preset: %s | Cumulative reward: %f | Steps: %f' % (i + 1, num_episodes, chosen_preset, episode_reward, episode_step))

        # Write results to csv file
        if USE_NOVEL_PRESETS:
            results_path = './eval_results_novel.csv'
        else:
            results_path = './eval_results_train.csv'
        if not os.path.exists(results_path):
            with open(results_path, 'w') as f:
                f.write('experiment, mean_reward, max_reward, min_reward, mean_steps, max_steps, min_steps\n')
        with open(results_path, 'a') as f:
            f.write(f'{exp_name},{int(np.mean(ep_rewards))},{int(np.max(ep_rewards))},{int(np.min(ep_rewards))},{int(np.mean(ep_steps))},{int(np.max(ep_steps))},{int(np.min(ep_steps))}\n')
        
        return ep_rewards, ep_steps

def make_env(args, weather_presets):

    # Initialize the CARLA environment
    env = carla_env.CarlaEnv(args.carla_town, args.max_npc_vehicles, 
                   args.desired_speed, args.max_stall_time, args.stall_speed, args.seconds_per_episode,
                   args.fps, 2000, 8000, args.env_verbose, args.camera_image_height, args.camera_image_width, 
                   args.fov, args.cam_x, args.cam_y, args.cam_z, args.cam_pitch,
                   args.lambda_r1, args.lambda_r2, args.lambda_r3, args.lambda_r4, args.lambda_r5,
                   weather_presets=weather_presets)
    
    # Set the random seed and reset
    env.seed(args.seed)
    env.reset()

    # Wrap CarlaEnv in FrameStack class to stack several consecutive frames together
    env = utils.FrameStack(env, k=args.frame_stack)

    return env

def main():

    # Parse arguments
    args = parse_args()
    verbose = False
    if args.env_verbose:
        verbose = True
    with open(os.path.join(args.experiment_dir_path, 'args.json'), 'r') as f:
        args.__dict__.update(json.load(f))
    if verbose: args.env_verbose = True

    # Set a fixed random seed for fair comparison across experiments.
    # The random seed 0 is sure to be an unused seed because the models
    # randomly select a seed from 1 to 1,000,000 in train.py.
    args.seed = 0

    # Random seed
    utils.set_seed_everywhere(args.seed)

    # Anchor/target data augmentor
    camera_image_shape = (args.camera_image_height, args.camera_image_width)
    augmentor = make_augmentor(args.augmentation, camera_image_shape)

    # In the evaluation, only novel weather presets are used in order 
    # to test the generalization/robustness capabilities of the agent.
    env = make_env(args, list(WEATHER_PRESETS.values()))

    # Shapes
    action_shape = env.action_space.shape
    pre_aug_obs_shape = env.observation_space.shape
    obs_shape = (3*args.frame_stack, args.augmented_image_height, args.augmented_image_width)

    # Make use of GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up agent
    agent = make_agent(obs_shape, action_shape, args, device, augmentor)

    # Load model
    model_dir_path = os.path.join(args.experiment_dir_path, 'model')
    agent.load(model_dir_path, str(args.augmentation), str(args.model_step))

    # Run evaluation loop
    ep_rewards, ep_steps = run_eval_loop(env, agent, augmentor, args.model_step, args.experiment_dir_path, num_episodes=50, record_video=False)

    # Deactivate the environment
    env.deactivate()

    # Print results
    print()
    print('Average reward: %f' % np.mean(ep_rewards))
    print('Max reward: %f' % np.max(ep_rewards))
    print('Min reward: %f' % np.min(ep_rewards))
    print('Std reward: %f' % np.std(ep_rewards))
    print()
    print('Average steps: %f' % np.mean(ep_steps))
    print('Max steps: %f' % np.max(ep_steps))
    print('Min steps: %f' % np.min(ep_steps))
    print('Std steps: %f' % np.std(ep_steps))

if __name__ == "__main__":
    main()