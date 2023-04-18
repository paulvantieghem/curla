# This piece of code was copied/modified from the following source:
#
#    Title: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning
#    Author: Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter
#    Date: 2020
#    Availability: https://github.com/MishaLaskin/curl

import numpy as np
import torch
import argparse
import os
import math
import gymnasium as gym
import sys
import random
import time
import json
import copy

import utils
from logger import Logger
from video import VideoRecorder
from torchvision import transforms

from curl_sac import CurlSacAgent
from carla_env import CarlaEnv


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Carla environment settings
    parser.add_argument('--carla_town', default='Town04', type=str)
    parser.add_argument('--max_npc_vehicles', default=100, type=int)
    parser.add_argument('--npc_ignore_traffic_lights_prob', default=0, type=int)
    parser.add_argument('--desired_speed', default=90, type=int) # km/h
    parser.add_argument('--max_stall_time', default=5, type=int) # seconds
    parser.add_argument('--stall_speed', default=0.5, type=float) # km/h
    parser.add_argument('--seconds_per_episode', default=100, type=int) # seconds
    parser.add_argument('--fps', default=20, type=int) # Hz

    # Carla camera settings
    parser.add_argument('--pre_transform_image_height', default=90, type=int)
    parser.add_argument('--pre_transform_image_width', default=160, type=int)
    parser.add_argument('--fov', default=120, type=int) # degrees
    parser.add_argument('--cam_x', default=1.3, type=float) # meters
    parser.add_argument('--cam_y', default=0.0, type=float) # meters
    parser.add_argument('--cam_z', default=1.75, type=float) # meters
    parser.add_argument('--cam_pitch', default=-15, type=int) # degrees

    # Carla reward function settings
    parser.add_argument('--lambda_r1', default=1.0, type=float) # Highway progression
    parser.add_argument('--lambda_r2', default=1.0, type=float) # Center of lane deviation
    parser.add_argument('--lambda_r3', default=1.0, type=float) # Steering angle
    parser.add_argument('--lambda_r4', default=1e-2, type=float) # Collision
    parser.add_argument('--lambda_r5', default=2.0, type=float) # Speeding
    parser.add_argument('--lambda_r6', default=1e2, type=float) # Solid lane marking crossing

    # Image augmentation settings
    parser.add_argument('--image_height', default=76, type=int)
    parser.add_argument('--image_width', default=135, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)

    # Replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100_000, type=int)

    # Train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1_000_000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)

    # Eval
    parser.add_argument('--eval_freq', default=25_000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)

    # Encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)

    # Actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # Critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above

    # SAC
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    
    # Misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='./tmp', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args

def run_eval_loop(env, agent, video, num_episodes, L, step, args, sample_stochastically=False):
        all_ep_rewards = []
        all_ep_steps = []
        all_ep_infos = {'r1': [], 'r2': [], 'r3': [], 'r4': [], 'r5': [], 'r6': [],
                            'mean_kmh': [], 'max_kmh': [], 'lane_crossing_counter': [], 'brake_sum': []}
        best_episode = {'reward': -math.inf, 'ep': -1}
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(True))
            done = False
            info = None
            episode_reward = 0
            episode_steps = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, (args.image_height, args.image_width))
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                video.record(env)
                episode_reward += reward
                episode_steps += 1

            if episode_reward > best_episode['reward']:
                best_episode['reward'] = episode_reward
                best_episode['ep'] = i

            video.save(f'eval_at_step_{step}_ep_{i+1}.mp4')
            all_ep_rewards.append(episode_reward)
            all_ep_steps.append(episode_steps)
            for k, v in info.items():
                all_ep_infos[k].append(v)

        bad_idx = list(range(num_episodes))
        bad_idx.remove(best_episode['ep'])
        for ep in bad_idx:
            os.remove(os.path.join(video_dir, f'eval_at_step_{step}_ep_{ep+1}.mp4'))
        
        # Log evaluation metrics
        L.log('eval/' + prefix + 'best_episode_reward', float(np.max(all_ep_rewards)), step)
        L.log('eval/' + prefix + 'mean_episode_reward', float(np.mean(all_ep_rewards)), step)
        L.log('eval/' + prefix + 'std_episode_reward', float(np.std(all_ep_rewards)), step)
        L.log('eval/' + prefix + 'best_episode_step', float(np.max(all_ep_steps)), step)
        L.log('eval/' + prefix + 'mean_episode_steps', float(np.mean(all_ep_steps)), step)
        L.log('eval/' + prefix + 'std_episode_steps', float(np.std(all_ep_steps)), step)
        for k, v in all_ep_infos.items():
            L.log('eval/' + prefix + 'mean_episode_' + k, float(np.mean(v)), step)

        return L

def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim

        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():

    # Parse arguments
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)

    # Random seed
    utils.set_seed_everywhere(args.seed)

    # Make necessary directories
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    ts = time.gmtime() 
    ts = time.strftime("%m-%d-%H-%M-%S", ts)    
    env_name = args.carla_town
    exp_name = env_name + '-' + ts + '-im' + str(args.image_height) + 'x' + str(args.image_width) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = os.path.join(args.work_dir, exp_name)
    utils.make_dir(args.work_dir)
    global video_dir 
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    # Logger
    L = Logger(args.work_dir, use_tb=args.save_tb)

    # Carla environment
    env = CarlaEnv(args.carla_town, args.max_npc_vehicles, args.npc_ignore_traffic_lights_prob, 
                   args.desired_speed, args.max_stall_time, args.stall_speed, args.seconds_per_episode,
                   args.fps, args.pre_transform_image_height, args.pre_transform_image_width, args.fov,
                   args.cam_x, args.cam_y, args.cam_z, args.cam_pitch,
                   args.lambda_r1, args.lambda_r2, args.lambda_r3, args.lambda_r4, args.lambda_r5, args.lambda_r6)
    env.seed(args.seed)
    env.reset()
    action_shape = env.action_space.shape

    # Stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    # Video recorder
    vid_path = video_dir if args.save_video else None
    video = VideoRecorder(vid_path, env.fps)

    # Store used arguments for repeatability
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # Make use of GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encoder type and observation shapes
    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_height, args.image_width)
        pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_height, args.pre_transform_image_width)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    # Replay buffer
    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_shape=(args.image_height, args.image_width),
    )

    # Agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    # Initializations
    episode, episode_reward, done, info = 0, 0, True, None
    start_time = time.time()
    fps = 0
    max_episode_reward = (env.desired_speed/3.6)*env.dt*env._max_episode_steps

    for step in range(args.num_train_steps+1):

        # Evaluate agent periodically
        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            L = run_eval_loop(env, agent, video, args.num_eval_episodes, L, step, args, sample_stochastically=False)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)
            done = True

        # Reset if done
        if done:

            # Log episode stats
            if step > 0:
                max_score_achieved = episode_reward/max_episode_reward
                L.log('train/episode_steps', episode_step, step)
                L.log('train/episode_reward', episode_reward, step)
                L.log('train/episode_max_score_ratio', max_score_achieved, step)
                L.log('train/episode_mean_fps', fps, step)
                if info != None:
                    L.log('train/episode_r1_sum', info['r1'], step)
                    L.log('train/episode_r2_sum', info['r2'], step)
                    L.log('train/episode_r3_sum', info['r3'], step)
                    L.log('train/episode_r4_sum', info['r4'], step)
                    L.log('train/episode_r5_sum', info['r5'], step)
                    L.log('train/episode_r6_sum', info['r6'], step)
                    L.log('train/episode_mean_kmh', info['mean_kmh'], step)
                    L.log('train/episode_max_kmh', info['max_kmh'], step)
                    L.log('train/episode_lane_crossing_counter', info['lane_crossing_counter'], step)
                    L.log('train/episode_brake_sum', info['brake_sum'], step)

            # Dump log
            L.dump(step)

            # Reset episode and environment
            start_time = time.time()
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            # Log episode stats
            L.log('train/episode', episode, step)

        # Sample action from the agent
        if step < args.init_steps:
            # Sample a random action within the bounds of the action space
            action = env.action_space.sample()
        else:
            # Evaluate the policy (CURL model) based on the observation
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Update the weights of the CURL model
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take the environment step
        next_obs, reward, done, info = env.step(action)

        # Allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)

        # Administration
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        obs = next_obs
        episode_step += 1
        fps = round(episode_step / (time.time() - start_time), 2)

    # Clear the environment
    env.deactivate()


if __name__ == '__main__':

    # Torch multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    # GPU information
    print('-'*100)
    print('PyTorch version:', torch.__version__)
    print('CUDA availability:', torch.cuda.is_available())
    print('CUDA device count:', torch.cuda.device_count())
    print('CUDA current device:', torch.cuda.current_device())
    print('CUDA device name (0):', torch.cuda.get_device_name(0))
    print('-'*100)

    # Main loop
    main()
