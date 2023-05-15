# The code in this file is partially based on (but does not directly copy) the following sources:
# 
# - The examples provided with the CARLA repository: 
#   https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples
#
# - The 'Self-driving cars with Carla and Python' series by 'sentdex' on YouTube: 
#   https://www.youtube.com/playlist?list=PLQVvvaa0QuDeI12McNQdnTlWz9XlCa0uo
#
# - The 'Learning Invariant Representations for Reinforcement Learning without Reconstruction' 
#   paper by Zhang et al. (2020):
#   https://arxiv.org/abs/2006.10742
#
# - The 'deep_bisim4control' repository by 'facebookresearch' on GitHub:
#   https://github.com/facebookresearch/deep_bisim4control 


# Standard library
import os
import random
import time
import math
import queue
import shutil
import pkg_resources
import importlib

# Installed 
import carla
import numpy as np
import cv2
import gymnasium as gym
gym.logger.set_level(40) # Sets the gym logger in ERROR mode (will not mention warnings)

# Modules
import settings
from carla_handler import CarlaServer

# Constants
TIMEOUT = 30.0      # Time in seconds to wait on various things
RENDER_WIDTH = 1152 # This size only matters for the video rendering, should be divisible by 64
RENDER_HEIGHT = 640 # This size only matters for the video rendering, should be divisible by 64
CARLA_VERSION = pkg_resources.get_distribution('carla').version
SPAWN_HEIGHT = 0.5 if CARLA_VERSION == '0.9.14' else 0.2
G = 9.807 # Gravitational acceleration in m/s^2

class CarlaEnv:

    # Constants
    show_preview = settings.SHOW_PREVIEW
    save_imgs = settings.SAVE_IMGS
    enable_spectator = settings.SPECTATOR
    MAX_STEER = settings.MAX_STEER
    MAX_THROTTLE_BRAKE = settings.MAX_THROTTLE_BRAKE
    THROTTLE_BRAKE_OFFSET = settings.THROTTLE_BRAKE_OFFSET
    weather_presets = settings.WEATHER_PRESETS


    def __init__(self, carla_town='Town04', max_npc_vehicles=10, desired_speed=65, max_stall_time=5, 
                 stall_speed=0.5, seconds_per_episode=50, fps=20, server_port=2000, tm_port=8000, verbose=False, pre_transform_image_height=90, 
                 pre_transform_image_width=160, fov=120, cam_x=1.3, cam_y=0.0, cam_z=1.75, cam_pitch=-15, 
                 lambda_r1=1.0, lambda_r2=0.3, lambda_r3=1.0, lambda_r4=0.005, lambda_r5=1.0):

        # Set parameters
        self.carla_town = carla_town
        self.max_npc_vehicles = max_npc_vehicles
        self.desired_speed = desired_speed
        self.max_stall_time = max_stall_time
        self.stall_speed = stall_speed
        self.seconds_per_episode = seconds_per_episode
        self.fps = fps
        self.server_port = server_port
        self.tm_port = tm_port
        self.dt = 1.0/fps
        self.verbose = verbose
        self.im_height = pre_transform_image_height
        self.im_width = pre_transform_image_width
        self.fov = fov
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_z = cam_z
        self.cam_pitch = cam_pitch
        self.lambda_r1 = lambda_r1
        self.lambda_r2 = lambda_r2
        self.lambda_r3 = lambda_r3
        self.lambda_r4 = lambda_r4
        self.lambda_r5 = lambda_r5

        # Initial values
        self.obs = None

        # Print ports
        print(f'CARLA server port: {self.server_port}')
        print(f'CARLA traffic manager port: {self.tm_port}')

        # Server
        if os.name == "nt":
            self.server = CarlaServer(port=self.server_port, offscreen=False, sound=False)
        else:
            self.server = CarlaServer(port=self.server_port, offscreen=True, sound=True)
        self.server.launch(delay=20.0, retries=3)

        # Client
        self.client = carla.Client('localhost', self.server_port)
        self.timeout = self.client.set_timeout(TIMEOUT)

        # World
        self.world = self.client.load_world(carla_town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        if self.verbose: print('loaded town %s' % self.map)

        # Set fixed simulation step for synchronous mode
        self.world_settings = self.world.get_settings()
        self.world_settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.world_settings)

        # Administration
        self.reset_counter = 0  # Counts how many times the environment has been reset (episode counter)
        self.episode_step = 0   # Counts the amount of time steps taken within the current episode
        self.total_step = 0     # Counts the total amount of time steps
        self.actor_list = []
        self.collision_history = []

        # Blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Route settings
        map_config = settings.map_config
        spawn_point_info = map_config[self.carla_town]
        ego_config = spawn_point_info['ego_config']
        npc_config = spawn_point_info['npc_config']

        # Ego vehicle spawn points
        self.ego_vehicle_possible_transforms = []
        for i in range(len(ego_config['lanes'])):
            ego_vehicle_transform = self.map.get_waypoint_xodr(road_id=ego_config['road_id'], 
                                                               lane_id=ego_config['lanes'][i], 
                                                               s=ego_config['start_s']).transform
            ego_vehicle_transform.location.z += SPAWN_HEIGHT # To avoid collision with road when spawning
            self.ego_vehicle_possible_transforms.append(ego_vehicle_transform)

        # NPC vehicle spawn points
        self.npc_vehicle_possible_transforms = []
        for idx in range(len(npc_config['road_id'])):

            # Extract current npc config information
            road_id = npc_config['road_id'][idx]
            start_lanes = npc_config['lanes'][idx]
            start_s = npc_config['start_s'][idx]
            npc_spawn_horizon = npc_config['max_s'][idx]
            npc_spawn_spacing = npc_config['spacing'][idx]

            # Calculate possible spawn points distances
            distances = list(range(int(npc_spawn_horizon/npc_spawn_spacing+1)))
            distances = [x*npc_spawn_spacing for x in distances]
            if road_id == ego_config['road_id']:
                distances_to_remove = []
                for distance in distances: # Make sure that no NPC vehicles spawn right next (or on) the ego vehicle
                    if distance < start_s + npc_spawn_spacing and distance > start_s - npc_spawn_spacing:
                        distances_to_remove.append(distance)
                for distance in distances_to_remove: distances.remove(distance)
            assert len(distances)*len(start_lanes) > self.max_npc_vehicles, 'Not enough spawn points for the desired amount of NPC vehicles'

            # Calculate possible spawn points transforms
            for i in range(len(distances)):
                npc_s = distances[i]
                for j in range(len(start_lanes)):
                    start_lane = start_lanes[j]
                    npc_vehicle_transform = self.map.get_waypoint_xodr(road_id=road_id, lane_id=start_lane, s=npc_s).transform
                    npc_vehicle_transform.location.z += SPAWN_HEIGHT # To avoid collision with road when spawning
                    self.npc_vehicle_possible_transforms.append(npc_vehicle_transform)
        print(f'[carla_env.py] Found {len(self.npc_vehicle_possible_transforms)} possible NPC vehicle spawn points for given configuration.')

        # Ego vehicle settings
        self.ego_vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]

        # Camera sensor settings
        self.camera_sensor_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_sensor_bp.set_attribute('image_size_x', f'{RENDER_WIDTH}')
        self.camera_sensor_bp.set_attribute('image_size_y', f'{RENDER_HEIGHT}')
        self.camera_sensor_bp.set_attribute('fov', f'{self.fov}')
        self.camera_sensor_bp.set_attribute('sensor_tick', f'{self.dt}')
        # self.camera_sensor_bp.set_attribute('exposure_compensation', str(-0.5))
        self.camera_sensor_transform = carla.Transform(carla.Location(x=self.cam_x, y=self.cam_y, z=self.cam_z), carla.Rotation(pitch=self.cam_pitch))

        # Collision sensor settings
        self.collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')

        # Traffic manager
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.global_percentage_speed_difference(30)

        # Setup for NPC vehicles
        self.npc_vehicle_blueprints = []
        self.npc_vehicle_models = ['audi.a2', 'audi.etron', 'audi.tt', 'bmw.grandtourer', 'charger2020.charger2020', 
                                   'chargercop2020.chargercop2020', 'chevrolet.impala', 'citroen.c3', 
                                   'dodge_charger.police', 'jeep.wrangler_rubicon', 'lincoln.mkz2017', 'lincoln2020.mkz2020', 
                                   'mercedes-benz.coupe', 'mercedesccc.mercedesccc', 'mini.cooperst', 'mustang.mustang', 
                                   'nissan.micra', 'nissan.patrol', 'seat.leon', 'tesla.model3', 'toyota.prius']
        for vehicle in self.blueprint_library.filter('*vehicle*'):
            if any(model in vehicle.id for model in self.npc_vehicle_models):
                self.npc_vehicle_blueprints.append(vehicle)

        # Spectator
        self.spectator = self.world.get_spectator()

        # Calculate max episode steps
        assert type(self.seconds_per_episode) == int
        assert type(self.fps) == int
        self._max_episode_steps = int(self.seconds_per_episode*self.fps)

        # Save camera sensor images
        if self.save_imgs:
            if os.path.exists('_out'):
                shutil.rmtree('_out')
            os.mkdir('_out')

        # Set synchronous mode
        self.set_synchronous_mode(True)

    def reset(self):
        
        # Destroy all actors of the previous simulation
        self.destroy_all_actors()

        # Administration
        self.actor_list = []
        self.npc_vehicles_list = []
        self.collision_history = []
        self.starting_frame_number = None

        # Set random weather preset with a random sun azimuth angle between 30 and 330 degrees
        weather_preset = random.choice(self.weather_presets)
        weather_preset.sun_azimuth_angle = np.random.randint(30, 330)
        self.world.set_weather(weather_preset)

        # Spawn ego vehicle
        start_time = time.time()
        while True:
            try:
                self.ego_vehicle_transform = random.choice(self.ego_vehicle_possible_transforms)
                self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, self.ego_vehicle_transform)
                control_action = carla.VehicleControl(throttle=0.1, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False)
                self.ego_vehicle.apply_control(control_action)
                break
            except:
                time.sleep(0.05)
            if time.time() - start_time > TIMEOUT:
                raise Exception('Timeout while waiting for ego vehicle to spawn')
        self.actor_list.append(self.ego_vehicle)
        if self.verbose: print('created %s' % self.ego_vehicle.type_id)

        # Place spectator if required
        if self.enable_spectator:
            yaw = self.ego_vehicle_transform.rotation.yaw*(math.pi/180)
            dist = -7.5
            dx = dist*math.cos(yaw)
            dy = dist*math.sin(yaw)
            self.spectator.set_transform(carla.Transform(self.ego_vehicle_transform.location + carla.Location(x=dx, y=dy, z=5), 
                                                         carla.Rotation(yaw=self.ego_vehicle_transform.rotation.yaw, pitch=-25)))

        # Spawn NPC vehicles
        npc_counter = 0
        start_time = time.time()
        batch = []
        while npc_counter < self.max_npc_vehicles and time.time() - start_time < 5.0:
            spawn_point_transform = random.choice(self.npc_vehicle_possible_transforms)
            temp = self.world.try_spawn_actor(random.choice(self.npc_vehicle_blueprints), spawn_point_transform)
            if temp is not None:
                self.npc_vehicles_list.append(temp)
                self.actor_list.append(temp)
                batch.append(carla.command.SetAutopilot(temp, True))
                npc_counter += 1
            else:
                if self.verbose: print('failed to spawn npc vehicle')
                time.sleep(0.01)
        if self.verbose: print(f'spawned {npc_counter} out of {self.max_npc_vehicles} npc vehicles')

        # Make sure that all vehicles fell down to the ground after spawning
        delta_t = math.sqrt((2*SPAWN_HEIGHT)/G) + 0.75   # Time to fall to the ground in seconds (ignoring air resistance) + margin
        nb_steps = math.ceil(delta_t/self.dt)           # Amount of steps in delta_t seconds (rounded up)
        for _ in range(nb_steps):
            self.world.tick(TIMEOUT)
            time.sleep(2*self.dt)

        # Set autopilot for all NPC vehicles
        self.client.apply_batch_sync(batch)

        # Spawn RGB camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_sensor_bp, self.camera_sensor_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor_queue = queue.Queue()
        self.camera_sensor.listen(self.camera_sensor_queue.put)
        if self.verbose: print('created %s' % self.camera_sensor.type_id)

        # Spawn collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.process_collision_data(event))
        if self.verbose: print('created %s' % self.collision_sensor.type_id)

        # Collect the initial sensor data and return the first frame number
        start_time = time.time()
        while True:
            try:
                self.starting_frame_number = self.collect_sensor_data()
                self.world.tick(TIMEOUT)
                break
            except:
                self.world.tick(TIMEOUT)
                time.sleep(2*self.dt)
                if self.verbose: print('ticking world after failed sensor data collection')
            if time.time() - start_time > TIMEOUT:
                raise Exception('Timeout while waiting for initial sensor data')

        # Administration
        self.reset_counter += 1
        self.episode_step = 0
        self.stall_counter = 0
        self.abs_kmh = 0.0
        if self.verbose: print('episode started')

        return self.obs
    
    def _process_action(self, action):

        # Process action
        action[0] = np.clip(action[0], -self.MAX_THROTTLE_BRAKE, self.MAX_THROTTLE_BRAKE)
        action[0] = np.clip(action[0] + self.THROTTLE_BRAKE_OFFSET, -self.MAX_THROTTLE_BRAKE, self.MAX_THROTTLE_BRAKE)
        action[1] = np.clip(action[1], -self.MAX_STEER, self.MAX_STEER)

        # Convert action to throttle, brake and steer
        throttle = float(np.max([action[0], 0.0]))
        brake = float(-np.min([action[0]/(1-self.THROTTLE_BRAKE_OFFSET), 0.0]))
        steer = float(action[1])
        
        return action, throttle, brake, steer

    def step(self, action):

        # Set traffic lights ahead of ego vehicle to green
        if self.ego_vehicle.is_at_traffic_light():
            traffic_light = self.ego_vehicle.get_traffic_light()
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.set_green_time(5.0)
            if self.verbose: print('traffic light ahead of ego vehicle set to green')

        # Apply the action to the ego vehicle
        action, self.throttle, self.brake, self.steer = self._process_action(action)
        control_action = carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=self.brake, hand_brake=False, reverse=False, manual_gear_shift=False)
        self.ego_vehicle.apply_control(control_action)

        # Calculate reward
        reward, done, info = self.reward_function(action)

        # Maximum episode time check
        if self.episode_step*self.dt + self.dt >= self.seconds_per_episode:
            done = True
            if self.verbose: print('episode done: episode time is up')

        # Tick world
        self.world.tick(TIMEOUT)
        self.episode_step += 1
        self.total_step += 1

        # Collect sensor data and 
        current_frame_number = self.collect_sensor_data()

        # This fixes a frame skip when the environment is reset
        if self.episode_step == 1:
            self.starting_frame_number = current_frame_number - 1

        return self.obs, reward, done, info

    def reward_function(self, action):

        # Initialize return information
        done = False
        reward = 0.0

        # Initializations
        if self.episode_step == 0:
            self.total_rewards = {'r1': 0.0, 'r2': 0.0, 'r3': 0.0, 'r4': 0.0, 'r5': 0.0}
            self.kmh_tracker = [0.0,]
            self.brake_sum = 0.0

        # Brake logging
        if action[0] < 0.0:
            self.brake_sum += -action[0]

        # Precision of the reward values
        precision = 4

        # Update waypoints
        p_prev_wp, p_next_wp = self._get_waypoints(distance=1.0)

        # Velocity vector of the ego vehicle
        self.v_ego = self.ego_vehicle.get_velocity()
        self.abs_kmh = float(3.6*math.sqrt(self.v_ego.x**2 + self.v_ego.y**2))
        self.v_ego = np.array([self.v_ego.x, self.v_ego.y])

        # Highway lane direction unit vector
        self.u_highway = p_next_wp - p_prev_wp
        norm = np.linalg.norm(self.u_highway)
        if np.isclose(norm, 0.0):
            self.u_highway = np.array([0.0, 0.0])
        else:
            self.u_highway = self.u_highway/norm

        # Reward for the highway progression [in meters] during the current time step
        r1 = self.lambda_r1*(np.dot(self.v_ego.T, self.u_highway)*self.dt)
        r1 = np.round(r1, precision)

        # Reward for perpendicular distance to the center of the lane [in meters] during the current time step,
        # smoothed to penalize small distances less and rounded to neglect really small distances
        distance = self._distance_from_center_lane(self.ego_vehicle, p_prev_wp, p_next_wp)
        r2 = (-1.0)*self.lambda_r2*np.round(np.minimum(1.0, distance**3), 2)
        r2 = np.round(r2, precision)

        # Reward for the current steering angle
        r3 = (-1.0)*self.lambda_r3*np.abs(self.steer)
        r3 = np.round(r3, precision)

        # Reward for collision intensities during the current time step
        r4 = 0.0
        if len(self.collision_history) != 0:
            intensities = []
            for collision in self.collision_history:
                # Wait to be at the correct frame to apply penalty
                if self.episode_step + self.starting_frame_number == collision.frame:
                    impulse = collision.normal_impulse
                    impulse = np.array([impulse.x, impulse.y, impulse.z])
                    intensities.append(np.linalg.norm(impulse))
            if len(intensities) > 0:
                intensities = np.array(intensities)
                r4 = (-1.0)*self.lambda_r4*np.sum(intensities)
                r4 = np.round(r4, precision)
                r4 = np.minimum(-25.0, r4)
                done = True
                if self.verbose: print('collision event: ', r4)

        # Reward for speeding during the current time step
        r5 = 0.0
        if self.abs_kmh > self.desired_speed + 1.0:
            velocity_delta = np.abs(self.abs_kmh - self.desired_speed)/3.6 # [m/s]
            # This ensures that the r5 punishment for speeding is greater than
            # the potential r1 reward for speeding (in straight line, see r1)
            r5 = self.dt*velocity_delta + self.dt
            r5 = (-1.0)*self.lambda_r5*r5
            r5 = np.round(r5, precision)

        # Total reward 
        if self.episode_step > 0:
            reward = r1 + r2 + r3 + r4 + r5

        # Update stalling counter
        if self.episode_step >= 50:
            if self.abs_kmh < self.stall_speed:
                self.stall_counter += 1
            else:
                self.stall_counter = 0

        # Terminate episode if stalling too long
        if self.stall_counter*self.dt >= self.max_stall_time:
            done = True
            if self.verbose: print('episode done: maximum stall time exceeded')

        # Extra information
        self.total_rewards['r1'] += r1 # Reward for highway progression
        self.total_rewards['r2'] += r2 # Penalty for center of lane deviation (--u-- shaped)
        self.total_rewards['r3'] += r3 # Penalty for the norm (absolute value) of the steering angle
        self.total_rewards['r4'] += r4 # Penalty for collision intensities
        self.total_rewards['r5'] += r5 # Penalty for speeding
        self.kmh_tracker.append(self.abs_kmh)
        self.info = {'r1': self.total_rewards['r1'], 
                'r2': self.total_rewards['r2'], 
                'r3': self.total_rewards['r3'], 
                'r4': self.total_rewards['r4'], 
                'r5': self.total_rewards['r5'], 
                'mean_kmh': np.mean(self.kmh_tracker), 
                'max_kmh': np.max(self.kmh_tracker), 
                'brake_sum': self.brake_sum}
        
        return reward, done, self.info
    
    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=0.0, high=255.0, shape=(3, self.im_height, self.im_width), dtype=np.uint8)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        return gym.spaces.Box(low=np.array([-self.MAX_THROTTLE_BRAKE, -self.MAX_STEER], dtype=np.float32), 
                              high=np.array([self.MAX_THROTTLE_BRAKE, self.MAX_STEER], dtype=np.float32), 
                              dtype=np.float32)
    
    def _get_waypoints(self, distance):
        """Returns the previous and next waypoints at a given distance from the ego vehicle."""
        waypoint = self.map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        previous_waypoint = waypoint.previous(distance)[0].transform.location
        next_waypoint = waypoint.next(distance)[0].transform.location
        p_prev_wp = np.array([previous_waypoint.x, previous_waypoint.y])
        p_next_wp = np.array([next_waypoint.x, next_waypoint.y])
        return p_prev_wp, p_next_wp
    
    def _distance_from_center_lane(self, vehicle, p_prev_wp, p_next_wp): 
        """Returns the perpendicular distance from the center of the lane."""
        p_ego = np.array([vehicle.get_location().x, vehicle.get_location().y])
        distance = np.linalg.norm(np.cross(p_next_wp - p_prev_wp, p_prev_wp - p_ego))/np.linalg.norm(p_next_wp - p_prev_wp)
        return distance

    def set_synchronous_mode(self, synchronous):
        '''Set the simulation to synchronous or asynchronous mode.'''
        self.world_settings.synchronous_mode = synchronous
        if synchronous:
            self.world_settings.fixed_delta_seconds = self.dt
        self.traffic_manager.set_synchronous_mode(synchronous)
        self.world.apply_settings(self.world_settings)

    def process_camera_data(self, carla_im_data):
        '''
        Process the raw image data from the camera sensor.
        -- self.image: HD image used for rendering (saving video)
        -- self.obs: downscaled image used as input to the model
        '''

        # Extract image data
        raw_image = np.array(carla_im_data.raw_data)

        # Reshape image data to (H, W, X) format (X = BGRA)
        bgra_image = raw_image.reshape((RENDER_HEIGHT, RENDER_WIDTH, -1))

        # Remove alpha to obtain (H, W, C) image with C = BGR
        bgr_image = bgra_image[:, :, :3]

        # Convert image from BGR to RGB
        self.rgb_image = bgr_image[:, :, ::-1]

        # Downscale image to requested size
        self.obs = cv2.resize(self.rgb_image, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA)

        # Display/record if requested
        if self.show_preview:
            cv2.imshow('', self.obs[:, :, ::-1])
            cv2.waitKey(1)
        if self.save_imgs:
            cv2.imwrite(os.path.join('_out', f'im_{self.reset_counter}_{self.episode_step}.png'), self.obs[:, :, ::-1])

        # Reshape image to (C, H, W) format required by the CURL model
        self.obs = np.transpose(self.obs, (2, 0, 1))

        if self.save_imgs:
            np.save(os.path.join('_out', f'im_{self.reset_counter}_{self.episode_step}.npy'), self.obs)
    
    def process_collision_data(self, event):
        '''Process the collision data from the collision sensor.'''
        self.collision_history.append(event)

    def collect_sensor_data(self):
        '''Collect the data from the camera sensor.'''
        try:
            camera_sensor_data = self.camera_sensor_queue.get(timeout=2.0)
        except:
            raise Exception('Timeout while waiting for camera sensor data')
        self.process_camera_data(camera_sensor_data)
        return camera_sensor_data.frame # Return frame number

    def seed(self, seed):
        '''Set the seed for the random number generators.'''
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
    
    def destroy_all_actors(self):
        '''Destroy all actors in the environment.'''
        if self.verbose: print('destroying actors')
        try: 
            self.camera_sensor.destroy()
            self.collision_sensor.destroy()
        except:
            print('[carla_env.py] No sensors to destroy or error destroying sensors.')
        if len(self.actor_list) != 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        if self.verbose: print('done.\n\n')

    def deactivate(self):
        '''Clean up the environment before closing it.'''
        self.set_synchronous_mode(False)
        self.destroy_all_actors()
        self.server.kill()
    
    def render(self):
        """Renders the current state of the environment."""

        frame = self.rgb_image.copy()

        # Define the dimensions and position of the bar charts
        bar_width = 200
        bar_height = 20
        bar_x = 10
        throttle_y = 30
        brake_y = 60
        steering_y = 90
        bar_color = (49, 61, 92)
        text_settings = (cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Calculate the width of the bars based on the driving information
        throttle_width = int(bar_width * self.throttle/self.MAX_THROTTLE_BRAKE)
        brake_width = int(bar_width * self.brake/self.MAX_THROTTLE_BRAKE)
        steering_width = int(bar_width * (self.steer/self.MAX_STEER) / 2)

        # Draw the throttle bar
        cv2.rectangle(frame, (bar_x, throttle_y), (bar_x + throttle_width, throttle_y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (bar_x, throttle_y), (bar_x + bar_width, throttle_y + bar_height), bar_color, 2)

        # Draw the brake bar
        cv2.rectangle(frame, (bar_x, brake_y), (bar_x + brake_width, brake_y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (bar_x, brake_y), (bar_x + bar_width, brake_y + bar_height), bar_color, 2)

        # Draw the steering bar
        if self.steer > 0:
            cv2.rectangle(frame, (bar_x + int(bar_width/2), steering_y), (bar_x + int(bar_width/2) + steering_width, steering_y + bar_height), bar_color, -1)
        else:
            cv2.rectangle(frame, (bar_x + int(bar_width/2) + steering_width, steering_y), (bar_x + int(bar_width/2), steering_y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (bar_x, steering_y), (bar_x + bar_width, steering_y + bar_height), bar_color, 2)
        cv2.rectangle(frame, (bar_x + int(bar_width/2) - 1, steering_y - 1), (bar_x + int(bar_width/2) + 1, steering_y + bar_height + 1), (255, 255, 255), -1)

        # Add the driving information to the frame as text
        cv2.putText(frame, 'Throttle', (bar_x + bar_width + 10, throttle_y + bar_height - 3), *text_settings)
        cv2.putText(frame, 'Brake',    (bar_x + bar_width + 10, brake_y + bar_height - 3),    *text_settings)
        cv2.putText(frame, 'Steering', (bar_x + bar_width + 10, steering_y + bar_height - 3), *text_settings)

        # Add highway and ego vehicle direction vectors to the frame
        if self.u_highway is not None and self.v_ego is not None:
            x = bar_x + int(bar_width/2)
            y = int(frame.shape[0]/2)
            factor = 75
            dx1 = int(self.u_highway[1]*factor)
            dy1 = int(self.u_highway[0]*factor)
            v_ego = self.v_ego/(self.desired_speed/3.6)
            dx2 = int(v_ego[1]*factor)
            dy2 = int(v_ego[0]*factor)
            cv2.arrowedLine(frame, (x, y), (x - dx1, y + dy1), bar_color, 2, cv2.LINE_AA)
            cv2.arrowedLine(frame, (x, y), (x - dx2, y + dy2), (255, 255, 255), 1, cv2.LINE_AA)

        # Add episode information to the frame as text
        if self.info is not None:
            x = frame.shape[1] - 170
            cv2.putText(frame, 'Cumulative reward', (x, 30), *text_settings)
            r1 = self.info['r1']
            cv2.putText(frame, f'r1: +{np.abs(r1):.4f}', (x, 60), *text_settings)
            r2 = self.info['r2']
            cv2.putText(frame, f'r2: -{np.abs(r2):.4f}', (x, 90), *text_settings)
            r3 = self.info['r3']
            cv2.putText(frame, f'r3: -{np.abs(r3):.4f}', (x, 120), *text_settings)
            r4 = self.info['r4']
            cv2.putText(frame, f'r4: -{np.abs(r4):.4f}', (x, 150), *text_settings)
            r5 = self.info['r5']
            cv2.putText(frame, f'r5: -{np.abs(r5):.4f}', (x, 180), *text_settings)
            r = r1 + r2 + r3 + r4 + r5
            cv2.putText(frame, f'Total: {r:.1f}', (x, 210), *text_settings)
            cv2.putText(frame, '-------------', (x, 240), *text_settings)
            mean_kmh = self.info['mean_kmh']
            cv2.putText(frame, f'Mean km/h: {mean_kmh:.1f}', (x, 270), *text_settings)
            max_kmh = self.info['max_kmh']
            cv2.putText(frame, f'Max km/h:  {max_kmh:.1f}', (x, 300), *text_settings)
            cv2.putText(frame, f'Cur km/h:  {self.abs_kmh:.1f}', (x, 330), *text_settings)

        return frame
    
