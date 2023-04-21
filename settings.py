import os

# Verbosity
VERBOSE = False

# Display camera sensor with OpenCV
SHOW_PREVIEW = False
SAVE_IMGS = False

# Move spectator to ego vehicle spawn location
if os.name == "nt":
    SPECTATOR = True
else:
    SPECTATOR = False

# Carla map configuration
map_config = {
    'Town04': {'road_id': 38, 'start_s': 33.0, 'start_lanes': [-1, -2, -3, -4]},
    }