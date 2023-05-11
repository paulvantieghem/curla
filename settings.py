import os
import carla

# Carla spawn location configuration
# WARNING: These values were only tested for CARLA 0.9.8, you might have to change them for other versions
lanes = [-1, -2, -3, -4]
map_config = {
    'Town04': {
        'ego_config': {
            'road_id': 39,                      # Road id of the road to spawn on
            'lanes': lanes,                     # Possible lanes to spawn vehicle in
            'start_s': 55.0,                    # Longitudinal distance along the road to spawn ego vehicle at
        }, 
        'npc_config': {
            'road_id': [39, 40],                # Road id of the road to spawn on
            'lanes': [lanes, lanes],            # Possible lanes to spawn vehicle in
            'start_s': [35.0, 10.0],            # Longitudinal distance along the road to start spawning vehicles at
            'spacing': [10.0, 10.0],            # Spacing between vehicles in meters
            'max_s': [135.0, 115.0],            # Longitudinal distance along the road to stop spawning vehicles at
        }
    }
}

# Carla weather presets for norma training and evaluation
WEATHER_PRESETS =  [carla.WeatherParameters.ClearNoon,
                    carla.WeatherParameters.ClearSunset, 
                    carla.WeatherParameters.CloudyNoon, 
                    carla.WeatherParameters.CloudySunset, 
                    carla.WeatherParameters.WetNoon, 
                    carla.WeatherParameters.WetSunset, 
                    carla.WeatherParameters.MidRainSunset]

# Carla weather presets for evaluation on unseen weather conditions
# WEATHER_PRESETS =  [carla.WeatherParameters.MidRainyNoon, ]

# Action space configuration
MAX_STEER = 0.3             # Number between 0.0 and 1.0
MAX_THROTTLE_BRAKE = 1.0    # Number between 0.0 and 1.0
THROTTLE_BRAKE_OFFSET = 0.0 # Number between 0.0 and 1.0
assert MAX_STEER > 0.0
assert MAX_THROTTLE_BRAKE > 0.0
assert THROTTLE_BRAKE_OFFSET >= 0.0
assert THROTTLE_BRAKE_OFFSET <= 0.8

#########################################
######## FOR DEBUGGING PURPOSES #########
#########################################

# Display the rgb camera images with OpenCV
SHOW_PREVIEW = False

# Save the rgb camera images to the _out folder
SAVE_IMGS = False

# Move spectator to ego vehicle spawn location
if os.name == "nt":
    SPECTATOR = True
else:
    SPECTATOR = False
