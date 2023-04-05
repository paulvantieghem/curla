import numpy as np
from carla_env import CarlaEnv
print('Test run of Carla environment')
env = CarlaEnv("Town04", 75, 10)
env.reset()
print('Carla environment is ready to go!')
for i in range(10):
    obs, reward, done, info = env.step(np.array([0.5, 0.0]))
    print('Step: {}, Reward: {}, Done: {}'.format(i, reward, done))
env.deactivate()
print(('Carla environment is deactivated!'))