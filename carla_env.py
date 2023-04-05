# Script based on 
# 
# - The examples provided with the CARLA repository: 
#   https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples
#
# - The 'Self-driving cars with Carla and Python' series by 'sentdex' on YouTube: 
#   https://www.youtube.com/playlist?list=PLQVvvaa0QuDeI12McNQdnTlWz9XlCa0uo

# Standard library
import os
import sys
import glob
import random
import time
import math
import queue
import shutil

# Installed 
try:
    sys.path.append(glob.glob(f'../carla/PythonAPI/carla/dist/carla-*{sys.version_info.major}.{sys.version_info.minor}-{"win-amd64" if os.name == "nt" else "linux-x86_64"}.egg')[0])
except IndexError:
    pass
import carla
import numpy as np
import cv2
import gymnasium as gym
gym.logger.set_level(40) # Sets the gym logger in ERROR mode (will not mention warnings)

# Modules
import settings

class CarlaEnv:

    # Constants
    show_preview = settings.SHOW_PREVIEW
    save_imgs = settings.SAVE_IMGS
    verbose = settings.VERBOSE
    enable_spectator = settings.SPECTATOR
    im_width = settings.IM_WIDTH
    im_height = settings.IM_HEIGHT
    fov = settings.FOV
    cam_x = settings.CAM_X
    cam_y = settings.CAM_Y
    cam_z = settings.CAM_Z
    seconds_per_episode = settings.SECONDS_PER_EPISODE
    fps = settings.FPS
    dt = 1.0/fps
    initial_speed = settings.INITIAL_SPEED

    # Initial values
    front_camera = None

    def __init__(self, carla_town, max_npc_vehicles, npc_ignore_traffic_lights_prob):

        # Set parameters
        self.carla_town = carla_town
        self.max_npc_vehicles = int(max_npc_vehicles)
        self.npc_ignore_traffic_lights_prob = int(npc_ignore_traffic_lights_prob)

        # Client
        self.client = carla.Client('localhost', 2000)
        self.timeout = self.client.set_timeout(20.0)

        # World
        self.world = self.client.load_world(carla_town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        if self.verbose: print('loaded town %s' % self.map)

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

        # Record the time of total steps and resetting steps
        self.reset_step = 0     # Counts how many times the environment has been reset (episode counter)
        self.episode_step = 0   # Counts the amount of time steps taken within the current episode
        self.total_step = 0     # Counts the total amount of time steps

        # Administration
        self.actor_list = []
        self.collision_history = []
        self.lane_invasion_history = []
        self.lane_invasion_len = 0

        # Blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Ego vehicle settings
        self.ego_vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]

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

        # Traffic manager
        self.traffic_manager = self.client.get_trafficmanager()

        # Setup for NPC vehicles
        self.npc_vehicle_spawn_points = self.world.get_map().get_spawn_points()
        self.npc_vehicle_models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
        self.npc_vehicle_blueprints = []
        for vehicle in self.blueprint_library.filter('*vehicle*'):
            if any(model in vehicle.id for model in self.npc_vehicle_models):
                self.npc_vehicle_blueprints.append(vehicle)
        self.max_npc_vehicles = min([self.max_npc_vehicles, len(self.npc_vehicle_spawn_points)])

        # Spectator
        self.spectator = self.world.get_spectator()

        # @TODO: investigate this
        self._max_episode_steps = self.seconds_per_episode*self.fps

        # Save camera sensor images
        if self.save_imgs:
            if os.path.exists('_out'):
                shutil.rmtree('_out')
            os.mkdir('_out')

    def reset(self):
        
        # Destroy all actors of the previous simulation
        self.destroy_all_actors()

        # Administration
        self.actor_list = []
        self.npc_vehicles_list = []
        self.collision_history = []
        self.lane_invasion_history = []
        self.lane_invasion_len = 0

        # Disable synchronous mode
        self.set_synchronous_mode(False)

        # Spawn ego vehicle
        while True:
            try:
                self.ego_vehicle_transform = random.choice(self.map.get_spawn_points())
                color = random.choice(self.ego_vehicle_bp.get_attribute('color').recommended_values)
                self.ego_vehicle_bp.set_attribute('color', color)
                self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, self.ego_vehicle_transform)
                break
            except:
                if self.verbose: print('Spawn failed because of collision at spawn position, trying again')
                time.sleep(0.01)
        self.actor_list.append(self.ego_vehicle)
        if self.verbose: print('created %s' % self.ego_vehicle.type_id)

        # Place spectator
        if self.enable_spectator:
            self.spectator.set_transform(carla.Transform(self.ego_vehicle_transform.location + carla.Location(z=75),carla.Rotation(pitch=-90)))

        # Set the initial speed to desired speed
        # yaw = (self.ego_vehicle.get_transform().rotation.yaw) * np.pi / 180.0
        # init_velocity = carla.Vector3D(
        #     x=self.initial_speed * np.cos(yaw),
        #     y=self.initial_speed * np.sin(yaw))
        # self.ego_vehicle.set_target_velocity(init_velocity)
        # time.sleep(2*self.dt) # Sleep for the duration of 2 frames in order for 'set_target_velocity' to take effect

        # Spawn RGB camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_sensor_bp, self.camera_sensor_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.camera_sensor)
        # self.camera_sensor.listen(lambda image: self.process_camera_data(image))
        self.camera_sensor_queue = queue.Queue()
        self.camera_sensor.listen(self.camera_sensor_queue.put)
        if self.verbose: print('created %s' % self.camera_sensor.type_id)

        # Spawn collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.process_collision_data(event))
        if self.verbose: print('created %s' % self.collision_sensor.type_id)

        # Spawn lane invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(self.lane_invasion_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.actor_list.append(self.lane_invasion_sensor)
        self.lane_invasion_sensor.listen(lambda event: self.process_lane_invasion_data(event))
        if self.verbose: print('created %s' % self.lane_invasion_sensor.type_id)

        # Spawn NPC vehicles
        for i, spawn_point in enumerate(random.sample(self.npc_vehicle_spawn_points, self.max_npc_vehicles)):
            temp = self.world.try_spawn_actor(random.choice(self.npc_vehicle_blueprints), spawn_point)
            if temp is not None:
                self.npc_vehicles_list.append(temp)
                self.actor_list.append(temp)

        # Parse the list of spawned NPC vehicles and give control to the TM through set_autopilot()
        for vehicle in self.npc_vehicles_list:
            vehicle.set_autopilot(True)
            # Randomly set the probability that a vehicle will ignore traffic lights
            self.traffic_manager.ignore_lights_percentage(vehicle, random.randint(0,self.npc_ignore_traffic_lights_prob))

        # Enable synchronous mode
        self.set_synchronous_mode(True)

        # Administration
        self.reset_step += 1
        self.episode_step = 0
        if self.verbose: print('episode started')

        # Collect initial data
        self.starting_frame = self.collect_sensor_data()

        return self.front_camera


    def step(self, action):

        # Apply the action to the ego vehicle
        if action[0]> 0:
            control_action = carla.VehicleControl(throttle=float(action[0]), steer=float(action[1]), brake=0)
        else:
            control_action = carla.VehicleControl(throttle=0, steer=float(action[1]), brake=-float(action[0]))
        # control_action = carla.VehicleControl(throttle=1.0, steer=0.0)
        self.ego_vehicle.apply_control(control_action)

        # Velocity conversion from vector in m/s to absolute speed in km/h
        v = self.ego_vehicle.get_velocity()
        abs_kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Initialize reward
        reward = 0
        done = False
        mistake = False

        # Reward for collision event
        if len(self.collision_history) != 0:
            if self.episode_step + self.starting_frame == self.collision_history[0].frame: # Wait to be at the correct frame to apply penalty
                if self.verbose: print('episode done: collision')
                done = True
                reward += -100
                mistake = True

        # Reward for  speed
        if abs_kmh > 1 and abs_kmh < 90:
            reward += abs_kmh/20
        else:
            reward += -2
        
        # Reward for lane invasion
        if self.lane_invasion_len < len(self.lane_invasion_history):
            lane_invasion_event = self.lane_invasion_history[-1]
            if self.episode_step + self.starting_frame == lane_invasion_event.frame: # Wait to be at the correct frame to apply penalty
                self.lane_invasion_len = len(self.lane_invasion_history)
                lane_markings = lane_invasion_event.crossed_lane_markings
                mistake = True
                for marking in lane_markings:
                    if str(marking.type) == 'Solid':
                        reward += -10
                    else:
                        reward += -5

        # Reward for the norm of the control actions
        # reward += -0.1*np.linalg.norm(action)**2

        # Reward for not making any mistakes
        if mistake == False:
            reward += 1

        # Maximum episode time check
        if self.episode_step*self.dt + self.dt >= self.seconds_per_episode:
            if self.verbose: print('episode done: episode time is up')
            done = True

        # Extra information
        extra_information = None

        # Tick world
        self.world.tick()
        self.episode_step += 1
        self.total_step += 1

        # Collect sensor data
        self.collect_sensor_data()

        return self.front_camera, reward, done, extra_information
    
    def render(self, mode):
        return self.image
    
    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        return gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
    
    def set_synchronous_mode(self, synchronous):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)
        self.traffic_manager.set_synchronous_mode(synchronous)

    def process_camera_data(self, carla_im_data):

        # Extract image data
        self.image = np.array(carla_im_data.raw_data)

        # Reshape image data to (H, W, X) format (X = channels + alpha)
        self.image = self.image.reshape((self.im_height, self.im_width, -1))

        # Remove alpha to obtain (H, W, C) image
        self.image = self.image[:, :, :3]

        # Display/record if requested
        if self.show_preview:
            cv2.imshow('', self.image)
            cv2.waitKey(1)
            time.sleep(0.2)
        if self.save_imgs:
            cv2.imwrite(os.path.join('_out', f'im_{self.reset_step}_{self.episode_step}.png'), self.image)

        # Reshape image to (C, H, W) format required by the CURL model
        self.front_camera = np.transpose(self.image, (2, 0, 1))
    
    def process_collision_data(self, event):
        self.collision_history.append(event)

    def process_lane_invasion_data(self, event):
        self.lane_invasion_history.append(event)

    def collect_sensor_data(self):
        camera_sensor_data = self.camera_sensor_queue.get(timeout=2.0)
        self.process_camera_data(camera_sensor_data)
        return camera_sensor_data.frame

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        self.traffic_manager.set_random_device_seed(0)
        return seed
    
    def destroy_all_actors(self):
        if self.verbose: print('destroying actors')
        if len(self.actor_list) != 0:
            try: 
                self.camera_sensor.destroy()
                self.lane_invasion_sensor.destroy()
                self.collision_sensor.destroy()
            except:
                pass
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        if self.verbose: print('done.')
        if self.verbose: print()
        if self.verbose: print()

    def deactivate(self):
        self.set_synchronous_mode(False)
        self.destroy_all_actors()
    
