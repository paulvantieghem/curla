import os

# Carla spawn location configuration
map_config = {
    'Town04': {
        'road_id': 38,                      # Road id of the road to spawn on
        'start_s': 45.0,                    # Longitudinal distance along the road to spawn ego vehicle at
        'start_lanes': [-1, -2, -3, -4],    # List of possible starting lane ids for both ego and NPC vehicles
        'npc_spawn_horizon': 225.0,         # Max distance ahead of the ego vehicle to spawn NPCs
        'npc_spawn_spacing': 5.0},          # Longitudinal spacing between NPC spawn points
    }

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
