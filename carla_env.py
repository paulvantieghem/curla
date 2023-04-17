# This code is based on (but does not directly copy) the following sources:
# 
# - The examples provided with the CARLA repository: 
#   https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples
#
# - The 'Self-driving cars with Carla and Python' series by 'sentdex' on YouTube: 
#   https://www.youtube.com/playlist?list=PLQVvvaa0QuDeI12McNQdnTlWz9XlCa0uo
#
# - The 'Learning Invariant Representations for Reinforcement Learning without Reconstruction' 
#   paper by Zhang et al. (2020) and 'deep_bisim4control' repository by 'facebookresearch' on GitHub:
#   https://arxiv.org/abs/2006.10742
#   https://github.com/facebookresearch/deep_bisim4control 
#   


# Standard library
import os
import random
import time
import math
import queue
import shutil

# Installed 
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


    def __init__(self, carla_town, max_npc_vehicles, npc_ignore_traffic_lights_prob, 
                   desired_speed, max_stall_time, stall_speed, seconds_per_episode,
                   fps, pre_transform_image_height, pre_transform_image_width, fov,
                   cam_x, cam_y, cam_z, cam_pitch, lambda_r1, lambda_r2, lambda_r3, 
                   lambda_r4, lambda_r5, lambda_r6):

        # Set parameters
        self.carla_town = carla_town
        self.max_npc_vehicles = max_npc_vehicles
        self.npc_ignore_traffic_lights_prob = npc_ignore_traffic_lights_prob
        self.desired_speed = desired_speed
        self.max_stall_time = max_stall_time
        self.stall_speed = stall_speed
        self.seconds_per_episode = seconds_per_episode
        self.fps = fps
        self.dt = 1.0/fps
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
        self.lambda_r6 = lambda_r6

        # Initial values
        self.front_camera = None

        # Client
        self.client = carla.Client('localhost', 2000)
        self.timeout = self.client.set_timeout(30.0)

        # World
        self.world = self.client.load_world(carla_town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        if self.verbose: print('loaded town %s' % self.map)

        # Weather
        self.weather_presets = [carla.WeatherParameters.ClearNoon, 
                                carla.WeatherParameters.ClearSunset, 
                                carla.WeatherParameters.CloudyNoon, 
                                carla.WeatherParameters.CloudySunset, 
                                carla.WeatherParameters.WetNoon, 
                                carla.WeatherParameters.WetSunset, 
                                carla.WeatherParameters.MidRainyNoon, 
                                carla.WeatherParameters.MidRainSunset]

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

        # Administration
        self.reset_step = 0     # Counts how many times the environment has been reset (episode counter)
        self.episode_step = 0   # Counts the amount of time steps taken within the current episode
        self.total_step = 0     # Counts the total amount of time steps
        self.actor_list = []
        self.collision_history = []
        self.lane_invasion_history = []
        self.lane_invasion_len = 0

        # Blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Highway spawn points
        if self.carla_town == 'Town04':
            map_config = settings.map_config
            self.highway_spawn_idx = list(map_config[self.carla_town]['spawn_ids']['highway'])
        else:
            print(f'[WARNING] No highway spawn points defined for {self.carla_town} in `settings.py`. Using all possible spawn points instead.')
            self.highway_spawn_idx = list(range(len(self.map.get_spawn_points())))

        # Ego vehicle settings
        self.ego_vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]

        # Camera sensor settings
        self.camera_sensor_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_sensor_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_sensor_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_sensor_bp.set_attribute('fov', f'{self.fov}')
        self.camera_sensor_bp.set_attribute('sensor_tick', f'{self.dt}')
        self.camera_sensor_bp.set_attribute('enable_postprocess_effects', str(True))
        self.camera_sensor_transform = carla.Transform(carla.Location(x=self.cam_x, y=self.cam_y, z=self.cam_z), carla.Rotation(pitch=self.cam_pitch))

        # Collision sensor settings
        self.collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')

        # Lane invasion sensor settings
        self.lane_invasion_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')

        # Traffic manager
        self.traffic_manager = self.client.get_trafficmanager()

        # Setup for NPC vehicles
        self.npc_vehicle_blueprints = ['audi', 'bmw', 'chevrolet', 'citroen', 'dodge', 'ford', 'jeep', 'lincoln', 'mercedes-benz', 'mini', 'nissan', 'seat', 'tesla', 'toyota', 'volkswagen']
        for vehicle in self.blueprint_library.filter('*vehicle*'):
            if any(model in vehicle.id for model in self.npc_vehicle_models):
                self.npc_vehicle_blueprints.append(vehicle)
        self.max_npc_vehicles = min([self.max_npc_vehicles, len(self.highway_spawn_idx)])

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

        # Set random weather
        self.world.set_weather(random.choice(self.weather_presets))

        # Spawn ego vehicle
        while True:
            try:
                id = random.choice(self.highway_spawn_idx)
                self.ego_vehicle_transform = self.map.get_spawn_points()[id]
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
            yaw = self.ego_vehicle_transform.rotation.yaw*(math.pi/180)
            dist = -7.5
            dx = dist*math.cos(yaw)
            dy = dist*math.sin(yaw)
            self.spectator.set_transform(carla.Transform(self.ego_vehicle_transform.location + carla.Location(x=dx, y=dy, z=5), 
                                                         carla.Rotation(yaw=self.ego_vehicle_transform.rotation.yaw, pitch=-25)))

        # Spawn NPC vehicles on highway
        npc_counter = 0
        for _ in range(self.max_npc_vehicles):
            id = random.choice(self.highway_spawn_idx)
            spawn_point_transform = self.map.get_spawn_points()[id]
            temp = self.world.try_spawn_actor(random.choice(self.npc_vehicle_blueprints), spawn_point_transform)
            if temp is not None:
                self.npc_vehicles_list.append(temp)
                self.actor_list.append(temp)
                npc_counter += 1
        if self.verbose: print(f'spawned {npc_counter} out of {self.max_npc_vehicles} npc vehicles')

        # Parse the list of spawned NPC vehicles and give control to the TM through set_autopilot()
        for vehicle in self.npc_vehicles_list:
            vehicle.set_autopilot(True)
            # Randomly set the probability that a vehicle will ignore traffic lights
            self.traffic_manager.ignore_lights_percentage(vehicle, random.randint(0,self.npc_ignore_traffic_lights_prob))

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

        # Spawn lane invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(self.lane_invasion_sensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.actor_list.append(self.lane_invasion_sensor)
        self.lane_invasion_sensor.listen(lambda event: self.process_lane_invasion_data(event))
        if self.verbose: print('created %s' % self.lane_invasion_sensor.type_id)

        # Make sure the ego vehicle is spawned in the center of the lane
        p_prev_wp, p_next_wp = self._get_waypoints(distance=1.0)
        dist = self._distance_from_center_lane(self.ego_vehicle, p_prev_wp, p_next_wp)
        if dist >= 1e-2:
            if self.verbose: print('Ego vehicle not spawned in center of the lane, resetting again...')
            time.sleep(1.0)
            self.reset()

        # Enable synchronous mode
        self.set_synchronous_mode(True)

        # Administration
        self.reset_step += 1
        self.episode_step = 0
        self.stall_counter = 0
        if self.verbose: print('episode started')

        # Collect initial data
        self.starting_frame = self.collect_sensor_data()

        return self.front_camera


    def step(self, action):

        # Apply the action to the ego vehicle
        action = np.clip(action, -1, 1)
        throttle = float(np.max([action[0], 0.0]))
        brake = float(-np.min([action[0], 0.0]))
        steer = float(action[1])
        control_action = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=False, reverse=False, manual_gear_shift=False)
        self.ego_vehicle.apply_control(control_action)

        # Calculate reward
        reward, done, info = self.reward_function(action)

        # Maximum episode time check
        if self.episode_step*self.dt + self.dt >= self.seconds_per_episode:
            done = True
            if self.verbose: print('episode done: episode time is up')

        # Tick world
        self.world.tick()
        self.episode_step += 1
        self.total_step += 1

        # Collect sensor data
        self.collect_sensor_data()

        return self.front_camera, reward, done, info

    def reward_function(self, action):

        # Initialize return information
        done = False
        reward = 0.0

        # Initializations
        if self.episode_step == 0:
            self.lane_invasion_len = 0
            self.total_rewards = {'r1': 0.0, 'r2': 0.0, 'r3': 0.0, 'r4': 0.0, 'r5': 0.0, 'r6': 0.0}
            self.kmh_tracker = [0.0,]
            self.lane_crossing_counter = 0

        # Precision of the reward values
        precision = 4

        # Update waypoints
        p_prev_wp, p_next_wp = self._get_waypoints(distance=1.0)

        # Velocity vector of the ego vehicle
        v_ego = self.ego_vehicle.get_velocity()
        abs_kmh = float(3.6*math.sqrt(v_ego.x**2 + v_ego.y**2))
        v_ego = np.array([v_ego.x, v_ego.y])

        # Highway lane direction unit vector
        u_highway = p_next_wp - p_prev_wp
        norm = np.linalg.norm(u_highway)
        if np.isclose(norm, 0.0):
            u_highway = np.array([0.0, 0.0])
        else:
            u_highway = u_highway/norm

        # Reward for the highway progression [in meters] during the current time step
        r1 = self.lambda_r1*(np.dot(v_ego.T, u_highway)*self.dt)
        r1 = np.round(r1, precision)

        # Reward for perpendicular distance to the center of the lane [in meters] during the current time step,
        # smoothed to penalize small distances less and rounded to neglect really small distances
        distance = self._distance_from_center_lane(self.ego_vehicle, p_prev_wp, p_next_wp)
        r2 = (-1.0)*self.lambda_r2*np.round(np.minimum(1.0, distance**3), 2)
        r2 = np.round(r2, precision)

        # Reward for the current steering angle
        steer_angle = action[1]
        r3 = (-1.0)*self.lambda_r3*np.abs(steer_angle)
        r3 = np.round(r3, precision)

        # Reward for collision intensities during the current time step
        r4 = 0.0
        if len(self.collision_history) != 0:
            intensities = []
            for collision in self.collision_history:
                # Wait to be at the correct frame to apply penalty
                if self.episode_step + self.starting_frame == collision.frame:
                    impulse = collision.normal_impulse
                    impulse = np.array([impulse.x, impulse.y, impulse.z])
                    intensities.append(np.linalg.norm(impulse))
            if len(intensities) > 0:
                intensities = np.array(intensities)
                r4 = (-1.0)*self.lambda_r4*np.sum(intensities)
                r4 = np.round(r4, precision)
                done = True
                if self.verbose: print('collision event: ', r4)

        # Reward for speeding during the current time step
        r5 = 0.0
        if abs_kmh > self.desired_speed + 1.0:
            velocity_delta = np.abs(abs_kmh - self.desired_speed)/3.6 # [m/s]
            # This ensures that the r5 punishment for speeding is greater than
            # the potential r1 reward for speeding (in straight line, see r1)
            r5 = self.dt*velocity_delta + self.dt
            r5 = (-1.0)*self.lambda_r5*r5
            r5 = np.round(r5, precision)

        # Reward for solid lane marking invasion
        r6 = 0.0
        if self.lane_invasion_len < len(self.lane_invasion_history):
            delta = len(self.lane_invasion_history) - self.lane_invasion_len
            lane_invasion_events = self.lane_invasion_history[-delta:]
            for lane_invasion_event in lane_invasion_events:
                if self.episode_step + self.starting_frame == lane_invasion_event.frame: # Wait to be at the correct frame to apply penalty
                    self.lane_invasion_len += 1
                    lane_markings = lane_invasion_event.crossed_lane_markings
                    for marking in lane_markings:
                        if str(marking.type) == 'Solid':
                            r6 = (-1.0)*self.lambda_r6
                            r6 = np.round(r6, precision)
                            done = True
                        elif str(marking.type) == 'Broken':
                            self.lane_crossing_counter += 1

        # Total reward 
        if self.episode_step > 0:
            reward = r1 + r2 + r3 + r4 + r5 + r6

        # Update stalling counter
        if self.episode_step >= 50:
            if abs_kmh < self.stall_speed:
                self.stall_counter += 1
            else:
                self.stall_counter = 0

        # Terminate episode if stalling too long
        if self.stall_counter*self.dt >= self.max_stall_time:
            done = True
            if self.verbose: print('episode done: maximum stall time exceeded')

        # Extra information
        self.total_rewards['r1'] += r1
        self.total_rewards['r2'] += r2
        self.total_rewards['r3'] += r3
        self.total_rewards['r4'] += r4
        self.total_rewards['r5'] += r5
        self.total_rewards['r6'] += r6
        self.kmh_tracker.append(abs_kmh)
        info = {'r1': self.total_rewards['r1'], 
                'r2': self.total_rewards['r2'], 
                'r3': self.total_rewards['r3'], 
                'r4': self.total_rewards['r4'], 
                'r5': self.total_rewards['r5'], 
                'r6': self.total_rewards['r6'],
                'mean_kmh': np.mean(self.kmh_tracker), 
                'max_kmh': np.max(self.kmh_tracker), 
                'lane_crossing_counter': self.lane_crossing_counter}
        
        return reward, done, info

    
    def render(self):
        return self.image
    
    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        return gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
    
    def _get_waypoints(self, distance):
        """Returns the previous and next waypoints at a given distance from the ego vehicle"""
        waypoint = self.map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        previous_waypoint = waypoint.previous(distance)[0].transform.location
        next_waypoint = waypoint.next(distance)[0].transform.location
        p_prev_wp = np.array([previous_waypoint.x, previous_waypoint.y])
        p_next_wp = np.array([next_waypoint.x, next_waypoint.y])
        return p_prev_wp, p_next_wp
    
    def _distance_from_center_lane(self, vehicle, p_prev_wp, p_next_wp): 
        """Returns the perpendicular distance from the center of the lane"""
        p_ego = np.array([vehicle.get_location().x, vehicle.get_location().y])
        distance = np.linalg.norm(np.cross(p_next_wp - p_prev_wp, p_prev_wp - p_ego))/np.linalg.norm(p_next_wp - p_prev_wp)
        return distance

    def set_synchronous_mode(self, synchronous):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)
        self.traffic_manager.set_synchronous_mode(synchronous)

    def process_camera_data(self, carla_im_data):

        # Extract image data
        self.image = np.array(carla_im_data.raw_data)

        # Reshape image data to (H, W, X) format (X = BGRA)
        self.image = self.image.reshape((self.im_height, self.im_width, -1))

        # Remove alpha to obtain (H, W, C) image wit C = BGR
        self.image = self.image[:, :, :3]

        # Convert image from BGR to RGB
        self.image = self.image[:, :, ::-1]

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
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        self.traffic_manager.set_random_device_seed(seed)
    
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
    
