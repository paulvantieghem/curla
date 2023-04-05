# Script based on 
# 
# - The examples provided with the CARLA repository: 
#   https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples
#
# - The 'Self-driving cars with Carla and Python' series by 'sentdex' on YouTube: 
#   https://www.youtube.com/playlist?list=PLQVvvaa0QuDeI12McNQdnTlWz9XlCa0uo

# Imports
import os
import sys
import glob
try:
    sys.path.append(glob.glob(f'../carla/PythonAPI/carla/dist/carla-*{sys.version_info.major}.{sys.version_info.minor}-{"win-amd64" if os.name == "nt" else "linux-x86_64"}.egg')[0])
except IndexError:
    pass
import carla
import random
import time
import math
import numpy as np
import cv2

# Constants
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
FOV = 110
CAM_X = 1.2
CAM_Y = 0.0
CAM_Z = 1.5
SECONDS_PER_EPISODE = 10

class CarlaEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    fov = FOV
    cam_x = CAM_X
    cam_y = CAM_Y
    cam_z = CAM_Z
    front_camera = None
    seconds_per_episode = SECONDS_PER_EPISODE

    def __init__(self) -> None:

        # Client
        self.client = carla.Client('localhost', 2000)
        self.timeout = carla.set_timeout(10.0)

        # World
        self.world = self.client.get_world()

        # Blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Ego vehicle settings
        self.ego_vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        color = random.choice(self.ego_vehicle_bp.get_attribute('color').recommended_values)
        self.ego_vehicle_bp.set_attribute('color', color)
        self.vehicle_transform = random.choice(self.world.get_map().get_spawn_points())

        # Sensor settings
        self.sensor_bp = self.blueprint_library.find('sensors.camera.rgb')
        self.sensor_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.sensor_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.sensor_bp.set_attribute('fov', f'{self.fov}')
        self.sensor_transform = carla.Transform(carla.Location(x=self.cam_x, y=self.cam_y, z=self.cam_z))

        # Collision sensor settings
        self.collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')

    def reset(self):

        # Administration
        self.collision_history = []
        self.actor_list = []

        # Spawn ego vehicle
        self.ego_vehicle = self.world.spawn_actor(self.model_3, self.vehicle_transform)
        self.actor_list.append(self.ego_vehicle)

        # Spawn sensor
        self.sensor = self.world.spawn_actor(self.sensor_bp, self.sensor_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda image: self.process_image(image))

        # Trigger ego vehicle with trivial input to activate it faster after spawning and wait for complete activation
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4.0)

        # Spawn collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # Wait for sensor to start
        while self.front_camera is None:
            time.sleep(0.01)

        # Start episode
        self.episode_start = time.time()
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera
    

    def collision_data(self, event):
        self.collision_history.append(event)
    

    def process_image(self, carla_im_data):
        '''
        Convert RGBA flat array to RGB numpy 3-channel array
        '''
        image = np.array(carla_im_data.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))
        image = image[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow('', image)
            cv2.waitKey(1)
        self.front_camera = image

    def step(self, action):
        if action == 0:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer = -1*self.STEER_AMT, brake=0.0))
        elif action == 1:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer = 0, brake=0.0))
        elif action == 2:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer = 1*self.STEER_AMT, brake=0.0))

        v = self.ego_vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_history != 0):
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + self.seconds_per_episode < time.time():
            done = True

        extra_information = None

        return self.front_camera, reward, done, extra_information