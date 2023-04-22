import os

# Verbosity of the CARLA environment
VERBOSE = True

# Display the rgb camera images with OpenCV
SHOW_PREVIEW = False

# Save the rgb camera images to the _out folder
SAVE_IMGS = False

# Move spectator to ego vehicle spawn location
if os.name == "nt":
    SPECTATOR = True
else:
    SPECTATOR = False

# Carla spawn location configuration
map_config = {
    'Town04': {'road_id': 38, 'start_s': 33.0, 'start_lanes': [-1, -2, -3, -4]},
    }
