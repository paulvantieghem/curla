import os
import time
import torch
import numpy as np

import utils
from curl_sac import CurlSacAgent
from carla_env import CarlaEnv
from video import VideoRecorder


def run_eval_loop(env, agent, step, num_episodes=10, encoder_type='pixel', img_shape=(76,135), record_video=False):
        
        # Initializations
        ep_rewards = []
        ep_times = []
        path = './eval_videos'
        if not os.path.exists(path):
            os.mkdir(path)
        video = VideoRecorder(path)

        # Run evaluation loop
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=record_video)
            done = False
            episode_reward = 0
            start_time = time.time()
            while not done:
                # center crop image
                if encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, (img_shape[0], img_shape[1]))
                with utils.eval_mode(agent):
                    action = agent.sample_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
            end_time = time.time()
            duration = end_time - start_time
            video.save('%d_%d.mp4' % step, i)
            ep_times.append(duration)
            ep_rewards.append(episode_reward)
            print('Episode %d/%d, Reward: %f, Time: %f' % (i + 1, num_episodes, episode_reward, duration))
        return ep_rewards, ep_times

def main():

    # Model checkpoint to load
    model_dir = './tmp/Town04-04-06-19-06-12-im76x135-b128-s312061-pixel/model'
    step = 20_000

    # Set up environment
    env = CarlaEnv('Town04', 75, 10)
    obs = env.reset()
    cam_obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    # Parameters
    frame_stack = 3
    cropped_shape = (int(np.ceil(0.84*cam_obs_shape[0])), int(np.ceil(0.84*cam_obs_shape[1])))
    obs_shape = (3*frame_stack, *cropped_shape)

    # Stack several consecutive frames together
    env = utils.FrameStack(env, k=frame_stack)

    # Make use of GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up agent
    agent = CurlSacAgent(obs_shape=obs_shape, action_shape=action_shape, device=device, encoder_type='pixel')

    # Load model
    agent.load_curl(model_dir, step)

    # Run evaluation loop
    ep_rewards, ep_times = run_eval_loop(env, agent, step, num_episodes=10, encoder_type='pixel', img_shape=cropped_shape, record_video=True)

    # Print results
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