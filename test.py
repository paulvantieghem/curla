from carla_env import CarlaEnv
import time
import numpy as np
import cv2

# Wait for Carla to be ready
while True:

    # Try to create environment. If fails, try again
    try:
        print('Trying to set up Carla environment...')
        env = CarlaEnv()
        break
    except:
        time.sleep(1)

for i in range(1):
    env.reset()
    total_reward = 0
    while True:
        action = np.array([0.8, 0.1])
        front_camera, reward, done, extra_information = env.step(action)
        total_reward += reward
        print(f'CURRENT STEP: reward: {reward}, done: {done}, extra_information: {extra_information}')
        print(f'TOTAL REWARD: {total_reward}')
        print()
        # cv2.imshow('', front_camera)
        # cv2.waitKey(1)
        time.sleep(0.05)
        if done:
            break

    # Clear all actors from simulator
    env.destroy_all_actors()
