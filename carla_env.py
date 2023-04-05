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
import settings
import cv2

class CarlaEnv:

    # Constants
    show_preview = settings.SHOW_PREVIEW
    im_width = settings.IM_WIDTH
    im_height = settings.IM_HEIGHT
    fov = settings.FOV
    cam_x = settings.CAM_X
    cam_y = settings.CAM_Y
    cam_z = settings.CAM_Z
    seconds_per_episode = settings.SECONDS_PER_EPISODE

    # Initial values
    front_camera = None

    def __init__(self, carla_town):

        # Client
        self.client = carla.Client('localhost', 2000)
        self.timeout = self.client.set_timeout(20.0)

        # World
        self.world = self.client.load_world(carla_town)
        self.world = self.client.get_world()
        print('loaded town %s' % carla_town)

        # Blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Ego vehicle settings
        self.ego_vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        color = random.choice(self.ego_vehicle_bp.get_attribute('color').recommended_values)
        self.ego_vehicle_bp.set_attribute('color', color)
        self.ego_vehicle_transform = random.choice(self.world.get_map().get_spawn_points())

        # Camera sensor settings
        self.camera_sensor_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_sensor_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_sensor_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_sensor_bp.set_attribute('fov', f'{self.fov}')
        self.camera_sensor_transform = carla.Transform(carla.Location(x=self.cam_x, y=self.cam_y, z=self.cam_z))

        # Collision sensor settings
        self.collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')

        # Lane invasion sensor settings
        self.lane_invasion_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')

        # Action space shape
        self.action_space = np.zeros((2,))
        self.observation_space = np.zeros((self.im_width, self.im_height, 3))
        self._max_episode_steps = 100  

    def reset(self):

        # Administration
        self.actor_list = []
        self.collision_history = []
        self.lane_invasion_history = []
        self.lane_invasion_len = 0

        # Spawn ego vehicle
        while True:
            try:
                self.ego_vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
                self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, self.ego_vehicle_transform)
                break
            except:
                print('Spawn failed because of collision at spawn position, trying again')
                time.sleep(0.01)
        self.actor_list.append(self.ego_vehicle)
        print('created %s' % self.ego_vehicle.type_id)

        # Trigger ego vehicle with trivial input to activate it faster after spawning and wait for complete activation
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4.0)

        # Spawn RGB camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_sensor_bp, self.camera_sensor_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda image: self.process_image(image))
        print('created %s' % self.camera_sensor.type_id)

        # Spawn collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        print('created %s' % self.collision_sensor.type_id)

        # Spawn lane invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(self.lane_invasion_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.actor_list.append(self.lane_invasion_sensor)
        self.lane_invasion_sensor.listen(lambda event: self.lane_invasion_data(event))
        print('created %s' % self.lane_invasion_sensor.type_id)


        # Wait for camera sensor to start receiving images
        while self.front_camera is None: # self.front_camera gets set to an image by the self.process_image function attached to the sensor
            time.sleep(0.01)
        print('%s is now receiving images' % self.camera_sensor.type_id)

        # Start episode
        self.episode_start = time.time()
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        print('episode started')

        return self.front_camera
    

    def collision_data(self, event):
        self.collision_history.append(event)

    def lane_invasion_data(self, event):
        self.lane_invasion_history.append(event)
    
    def process_image(self, carla_im_data):
        '''
        Convert RGBA flat array to RGB numpy 3-channel array
        '''
        image = np.array(carla_im_data.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))
        image = image[:, :, :3]
        if self.show_preview:
            cv2.imshow('', image)
            cv2.waitKey(1)
            time.sleep(0.1)
        self.front_camera = image.reshape((3, self.im_height, self.im_width))

    def step(self, action):

        # Apply the action to the ego vehicle
        if action[0]> 0:
            control_action = carla.VehicleControl(throttle=float(action[0]), steer=float(action[1]), brake=0)
        else:
            control_action = carla.VehicleControl(throttle=0, steer=float(action[1]), brake=-float(action[0]))
        self.ego_vehicle.apply_control(control_action)

        # Velocity conversion from vector in m/s to absolute speed in km/h
        v = self.ego_vehicle.get_velocity()
        abs_kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Calculate current reward
        if len(self.collision_history) != 0:
            print('EPISODE STOPPED: collision')
            done = True
            reward = -100
        elif abs_kmh < 10:
            done = False
            reward = -1
        elif self.lane_invasion_len < len(self.lane_invasion_history):
            lane_invasion_event = self.lane_invasion_history[-1]
            lane_markings = lane_invasion_event.crossed_lane_markings
            for marking in lane_markings:
                print(str(marking.type))
                if str(marking.type) == 'Solid':
                    reward = -10
                else:
                    reward = -5
            done = False
        else:
            done = False
            reward = 1

        # Maximum episode time check
        if self.episode_start + self.seconds_per_episode < time.time():
            print('EPISODE STOPPED: episode done')
            done = True

        # Extra information
        extra_information = None

        return self.front_camera, reward, done, extra_information
    
    def destroy_all_actors(self):
        print('destroying actors')
        self.camera_sensor.destroy()
        self.lane_invasion_sensor.destroy()
        self.collision_sensor.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print('done.')

    def seed(self, seed):
        random.seed(seed)
    
