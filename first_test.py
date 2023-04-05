import glob
import sys
import platform
import random
import time
import numpy as np

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
IM_H = 720
IM_W = 1280
FOV = 110

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
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # Set the color of the vehicle to a random recommended color
    color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
    vehicle_bp.set_attribute('color', color)

    # Now we need to give an initial transform to the vehicle. We choose a
    # random transform from the list of recommended spawn points of the map.
    transform = random.choice(world.get_map().get_spawn_points())

    # So let's tell the world to spawn the vehicle. It is important to note 
    # that the actors we create won't be destroyed unless we call their 
    # "destroy" function. If we fail to call "destroy" they will stay in 
    # the simulation even after we quit the Python script.
    # For that reason, we are storing all the actors we create so we can
    # destroy them afterwards.
    vehicle = world.spawn_actor(vehicle_bp, transform)
    actor_list.append(vehicle)
    location = vehicle.get_location()
    print('created %s' % vehicle.type_id, 'at location', location)

    # Let's put the vehicle to drive around.
    vehicle.set_autopilot(True)

    # Let's add now a "depth" camera attached to the vehicle. Note that the
    # transform we give here is now relative to the vehicle.
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{IM_W}')
    camera_bp.set_attribute('image_size_y', f'{IM_H}')
    camera_bp.set_attribute('fov', f'{FOV}')
    camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera)
    print('created %s' % camera.type_id)

    # Now we register the function that will be called each time the sensor
    # receives an image. In this example we are saving the image to disk.
    camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))

    time.sleep(10)

    # Turn off autopilot
    vehicle.set_autopilot(False)
    print('Autopilot off')

    # Apply random action
    input = np.random.uniform(low=-1.0, high=1.0, size=(2,)) # Simulates the output of a neural net for example
    print(input)
    if input[0] > 0:
        action = carla.VehicleControl(throttle=float(input[0]), steer=float(input[1]), brake=0)
    else:
        action = carla.VehicleControl(throttle=0, steer=float(input[1]), brake=-float(input[0]))
    vehicle.apply_control(action)
    
    time.sleep(10)

  finally:
      print('destroying actors')
      camera.destroy()
      client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
      print('done.')


if __name__ == '__main__':
    main()