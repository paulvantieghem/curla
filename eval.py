import os
import time
import argparse
import torch
import numpy as np
import random
import math

import utils
import augmentations
from curl_sac import CurlSacAgent
from carla_env import CarlaEnv
from video import VideoRecorder

def parse_args():
    parser = argparse.ArgumentParser()

    # Model specifications
    parser.add_argument('--model_dir_path', default='', type=str)
    parser.add_argument('--model_step', default=0, type=int)

    # Carla environment settings
    parser.add_argument('--carla_town', default='Town04', type=str)
    parser.add_argument('--max_npc_vehicles', default=10, type=int)
    parser.add_argument('--desired_speed', default=65, type=int) # km/h | NPCs drive at 70% of the speed limit (90), which is 63 km/h
    parser.add_argument('--max_stall_time', default=5, type=int) # seconds
    parser.add_argument('--stall_speed', default=0.5, type=float) # km/h
    parser.add_argument('--seconds_per_episode', default=50, type=int) # seconds
    parser.add_argument('--fps', default=20, type=int) # Hz
    parser.add_argument('--env_verbose', default=False, action='store_true') # Verbosity of the CARLA environment 'CarlaEnv' class

    # Carla camera settings
    parser.add_argument('--camera_image_height', default=90, type=int)
    parser.add_argument('--camera_image_width', default=160, type=int)
    parser.add_argument('--fov', default=120, type=int) # degrees
    parser.add_argument('--cam_x', default=1.3, type=float) # meters
    parser.add_argument('--cam_y', default=0.0, type=float) # meters
    parser.add_argument('--cam_z', default=1.75, type=float) # meters
    parser.add_argument('--cam_pitch', default=-15, type=int) # degrees

    # Carla reward function settings
    parser.add_argument('--lambda_r1', default=1.0, type=float)     # Highway progression
    parser.add_argument('--lambda_r2', default=0.3, type=float)     # Center of lane deviation
    parser.add_argument('--lambda_r3', default=1.0, type=float)     # Steering angle
    parser.add_argument('--lambda_r4', default=0.005, type=float)   # Collision
    parser.add_argument('--lambda_r5', default=1.0, type=float)     # Speeding

    # Image augmentation settings
    parser.add_argument('--augmentation', default='random_crop', type=str)
    parser.add_argument('--frame_stack', default=3, type=int)

    # Random seed
    parser.add_argument('--seed', default=-1, type=int)

    args = parser.parse_args()
    return args


def run_eval_loop(env, agent, augmentor, step, num_episodes=10, record_video=False):
        
        # Initializations
        ep_rewards = []
        ep_times = []
        path = './eval_videos'
        if not os.path.exists(path):
            os.mkdir(path)
        video = VideoRecorder(path, env.fps)

        # Run evaluation loop
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=record_video)
            done = False
            episode_reward = 0
            episode_step = 0
            start_time = time.time()
            while not done:

                # Perform anchor augmentation
                obs = augmentor.anchor_augmentation(obs)

                # Sample action from agent
                with utils.eval_mode(agent):
                    action = agent.sample_action(obs)

                # Take step in environment
                obs, reward, done, info = env.step(action)

                # Administration and logging
                video.record(env)
                episode_reward += reward
                episode_step += 1
                if info is not None and (episode_step % 20 == 0 or done):
                    print('-' * 50)
                    print('Step: %d' % episode_step)
                    print('Highway progression (r1): %f' % info['r1'])
                    print('Lane deviation (r2): %f' % info['r2'])
                    print('Steering (r3): %f' % info['r3'])
                    print('Collision (r4): %f' % info['r4'])
                    print('Speeding (r5): %f' % info['r5'])
                    print('Mean kmh: %f' % info['mean_kmh'])
                    print('Max kmh: %f' % info['max_kmh'])
                    
            end_time = time.time()
            duration = end_time - start_time
            video.save(f'{step}_{i}.mp4')
            ep_times.append(duration)
            ep_rewards.append(episode_reward)
            print('Episode %d/%d, Reward: %f, Time: %f' % (i + 1, num_episodes, episode_reward, duration))
        return ep_rewards, ep_times

def main():

    # Parse arguments
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)

    # Random seed
    utils.set_seed_everywhere(args.seed)

    # Augmentation class
    augmentor = None
    camera_image_shape = (args.camera_image_height, args.camera_image_width)
    if args.augmentation == 'identity':
        augmentor = augmentations.IdentityAugmentation(camera_image_shape)
    elif args.augmentation == 'random_crop':
        augmentor = augmentations.RandomCrop(camera_image_shape)
    else:
        raise ValueError('augmentation is not supported: %s' % args.augmentation)
    
    # Set the output shape of the augmentation
    args.augmented_image_height = augmentor.output_shape[0]
    args.augmented_image_width = augmentor.output_shape[1]

    # Set up environment
    env = CarlaEnv(args.carla_town, args.max_npc_vehicles,
                   args.desired_speed, args.max_stall_time, args.stall_speed, args.seconds_per_episode,
                   args.fps, args.env_verbose, args.camera_image_height, args.camera_image_width, 
                   args.fov, args.cam_x, args.cam_y, args.cam_z, args.cam_pitch,
                   args.lambda_r1, args.lambda_r2, args.lambda_r3, args.lambda_r4, args.lambda_r5)
    env.seed(args.seed) # Important not to remove !
    env.reset()

    # Shapes
    action_shape = env.action_space.shape
    pre_aug_obs_shape = env.observation_space.shape
    obs_shape = (3*args.frame_stack, args.augmented_image_height, args.augmented_image_width)

    # Stack several consecutive frames together
    env = utils.FrameStack(env, k=args.frame_stack)

    # Make use of GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up agent
    agent = CurlSacAgent(obs_shape=obs_shape, action_shape=action_shape, device=device, augmentor=augmentor, encoder_type=args.encoder_type)

    # Load model
    print('Loading model %s' % os.path.join(args.model_dir_path, 'curl_%d.pt' % args.model_step))
    agent.load_curl(args.model_dir_path, str(args.model_step))

    # Run evaluation loop
    ep_rewards, ep_times = run_eval_loop(env, agent, augmentor, args.model_step, num_episodes=1, record_video=True)

    # Deactivate the environment
    env.deactivate()

    # Print results
    print()
    print('Average reward: %f' % np.mean(ep_rewards))
    print('Max reward: %f' % np.max(ep_rewards))
    print('Min reward: %f' % np.min(ep_rewards))
    print('Std reward: %f' % np.std(ep_rewards))
    print()
    print('Average time: %f' % np.mean(ep_times))
    print('Max time: %f' % np.max(ep_times))
    print('Min time: %f' % np.min(ep_times))
    print('Std time: %f' % np.std(ep_times))

if __name__ == "__main__":
    main()