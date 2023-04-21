import carla
import settings
import numpy as np
import random
import math
import time

# Client
client = carla.Client('localhost', 2000)
timeout = client.set_timeout(30.0)

# World
carla_town = 'Town04'
world = client.load_world(carla_town)
world = client.get_world()
map = world.get_map()

blueprint_library = world.get_blueprint_library()
ego_vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

start_lane = random.choice([-1, -2 , -3, -4])
location = carla.Location(x=348.4, y=14.5)
waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
road_id = 38
s = 30.0
print('road_id: ', road_id)
print('s: ', s)
ego_vehicle_transform = map.get_waypoint_xodr(road_id=road_id, lane_id=start_lane, s=s).transform
print(ego_vehicle_transform.location.z)
ego_vehicle_transform.location.z += 2
print(ego_vehicle_transform.location.z)
world.debug.draw_string(ego_vehicle_transform.location, str(start_lane), draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=240.0, persistent_lines=True)
try:
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_vehicle_transform)
except:
    print('Failed to spawn ego vehicle')
    time.sleep(0.01)




spectator = world.get_spectator()
yaw = ego_vehicle_transform.rotation.yaw*(math.pi/180)
dist = -7.5
dx = dist*math.cos(yaw)
dy = dist*math.sin(yaw)
spectator.set_transform(carla.Transform(ego_vehicle_transform.location + carla.Location(x=dx, y=dy, z=5), 
                                                         carla.Rotation(yaw=ego_vehicle_transform.rotation.yaw, pitch=-25)))