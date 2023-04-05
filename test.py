print('\n[TEST 1: PYTORCH]')
import torch
print('----PyTorch version: {}'.format(torch.__version__))
print('----CUDA is available: {}'.format(torch.cuda.is_available()))
print('----CUDA version: {}'.format(torch.version.cuda))
print('----cudnn version: {}'.format(torch.backends.cudnn.version()))
print('----CUDA device count: {}'.format(torch.cuda.device_count()))
print('----CUDA device name: {}'.format(torch.cuda.get_device_name(0)))
print('----CUDA device capability: {}'.format(torch.cuda.get_device_capability(0)))
print('----CUDA device total memory: {}'.format(torch.cuda.get_device_properties(0).total_memory))
print('----CUDA device memory used: {}'.format(torch.cuda.memory_allocated(0)))
print('----CUDA device memory cached: {}'.format(torch.cuda.memory_cached(0)))
print('----CUDA device memory reserved: {}'.format(torch.cuda.memory_reserved(0)))
print('----CUDA device memory max reserved: {}'.format(torch.cuda.max_memory_reserved(0)))
print('----CUDA device memory max allocated: {}'.format(torch.cuda.max_memory_allocated(0)))
print('----CUDA device memory max cached: {}'.format(torch.cuda.max_memory_cached(0)))
print('----CUDA device memory max reserved: {}'.format(torch.cuda.max_memory_reserved(0)))

print('\n[TEST 2: CARLA]')
import numpy as np
from carla_env import CarlaEnv
env = CarlaEnv("Town04", 75, 10)
env.reset()
print('----Carla environment is ready to go!')
for i in range(10):
    obs, reward, done, info = env.step(np.array([0.5, 0.0]))
    print('----Step: {}, Reward: {}, Done: {}'.format(i, reward, done))
env.deactivate()
print(('----Carla environment is deactivated!'))