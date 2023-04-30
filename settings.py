import os

# Move spectator to ego vehicle spawn location
if os.name == "nt":
    SPECTATOR = True
else:
    SPECTATOR = False

# Carla spawn location configuration
map_config = {
    'Town04': {'road_id': 38, 'start_s': 33.0, 'start_lanes': [-1, -2, -3, -4], 'npc_spawn_horizon': 225.0, 'npc_spawn_spacing': 5.0},
    }

#########################################
######## FOR DEBUGGING PURPOSES #########
#########################################

# Display the rgb camera images with OpenCV
SHOW_PREVIEW = False

# Save the rgb camera images to the _out folder
SAVE_IMGS = False
