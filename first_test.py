# Script based on 
# 
# - The examples provided with the CARLA repository: 
#   https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples
#
# - The 'Self-driving cars with Carla and Python' series by 'sentdex' on YouTube: 
#   https://www.youtube.com/playlist?list=PLQVvvaa0QuDeI12McNQdnTlWz9XlCa0uo

import glob
import sys
import platform
import random
import time

import numpy as np
import cv2

# Add the carla library location to the system path for import
os_name = platform.system()
if os_name == 'Windows':
    attachment = 'win-amd64'
elif os_name == 'Linux':
    attachment = 'linux-x86_64'
else:
    raise Exception("System OS must be either Windows or Linux") 

try:
    sys.path.append(glob.glob('../carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        attachment))[0])
    import carla
except IndexError:
    raise Exception("Something went wrong when trying to add the carla library to the system path and/or importing carla")

# Parameters of the camera attached to the ego vehicle
IM_H = 540
IM_W = 960
FOV = 110

def process_image(carla_im_data):
    '''
    Convert RGBA flat array to RGB numpy 3-channel array
    '''
    image = np.array(carla_im_data.raw_data)
    image = image.reshape((carla_im_data.height, carla_im_data.width, -1))
    image = image[:, :, :3]
    cv2.imshow('', image)
    cv2.waitKey(1)
    return image/255.0 # Normalize data before passing it to a neural network!

def main():

  try:
      
    # Create an empty actor list, it is important to keep track of actors
    actor_list = []

    # Create a client that will send requests to the server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Get the world that is currently running
    world = client.get_world()

    # The world contains the list blueprints that we can use for adding new actors into the simulation.
    blueprint_library = world.get_blueprint_library()

    # Get the Tesla Model 3 blueprint from the blueprint library
    ego_vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # Set the color of the vehicle to a random recommended color
    color = random.choice(ego_vehicle_bp.get_attribute('color').recommended_values)
    ego_vehicle_bp.set_attribute('color', color)

    # Now we need to give an initial transform to the vehicle. We choose a
    # random transform from the list of recommended spawn points of the map.
    transform = random.choice(world.get_map().get_spawn_points())

    # So let's tell the world to spawn the vehicle. It is important to note 
    # that the actors we create won't be destroyed unless we call their 
    # "destroy" function. If we fail to call "destroy" they will stay in 
    # the simulation even after we quit the Python script.
    # For that reason, we are storing all the actors we create so we can
    # destroy them afterwards.
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)
    actor_list.append(ego_vehicle)
    location = ego_vehicle.get_location()
    print('created %s' % ego_vehicle.type_id, 'at location', location)

    # Let's put the vehicle to drive around.
    ego_vehicle.set_autopilot(True)

    # Let's add now a "depth" camera attached to the vehicle. Note that the
    # transform we give here is now relative to the vehicle.
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{IM_W}')
    camera_bp.set_attribute('image_size_y', f'{IM_H}')
    camera_bp.set_attribute('fov', f'{FOV}')
    camera_transform = carla.Transform(carla.Location(x=1.2, z=1.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    actor_list.append(camera)
    print('created %s' % camera.type_id)

    # Now we register the function that will be called each time the sensor
    # receives an image. In this example we are saving the image to disk.
    # camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))
    camera.listen(lambda image: process_image(image))

    time.sleep(10)

    # Turn off autopilot
    ego_vehicle.set_autopilot(False)
    print('Autopilot off')

    # Apply random action
    input = np.random.uniform(low=-1.0, high=1.0, size=(2,)) # Simulates the output of a neural net for example
    print('Input to be applied to the vehicle:', input)
    if input[0] > 0:
        action = carla.VehicleControl(throttle=float(input[0]), steer=float(input[1]), brake=0)
    else:
        action = carla.VehicleControl(throttle=0, steer=float(input[1]), brake=-float(input[0]))
    ego_vehicle.apply_control(action)
    
    time.sleep(10)

  finally:
      print('destroying actors')
      camera.destroy()
      client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
      print('done.')


if __name__ == '__main__':
    main()