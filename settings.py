# Verbosity
VERBOSE = False

# Move spectator to ego vehicle spawn location
SPECTATOR = True

# Display camera sensor with OpenCV
SHOW_PREVIEW = False
SAVE_IMGS = False

# Speeds in km/h
SET_INITIAL_SPEED = False
INITIAL_SPEED = 10  # km/h
DESIRED_SPEED = 90 # km/h

# Maximum stall time
MAX_STALL_TIME = 5
STALL_SPEED = 1

# Synchronous mode FPS
FPS = 20

# Episode settings
SECONDS_PER_EPISODE = 1000/FPS

# RGB camera settings
IM_HEIGHT = 90
IM_WIDTH = 160
FOV = 110

# RGB camera relative position
CAM_X = 1.5
CAM_Y = 0.0
CAM_Z = 1.75

# Carla map configuration
map_config = {
    "Town04": {
        "spawn_ids": {
            # Spawn points on town roads
            "town": {
                159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                180, 181, 182, 183, 192, 193, 194, 195, 230, 231, 232, 233, 234, 235, 236, 237, 245, 247, 248, 249, 250,
                251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 274, 275, 276, 277, 286, 287, 288, 307,
                308, 309, 311, 338, 339, 340, 341, 342, 343, 344, 361, 362, 363, 364
            },
            # Spawn points on highways
            "highway": {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
                127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 184, 185, 186, 187, 188, 189, 190, 191, 196, 197,
                198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 217, 218, 219,
                222, 223, 224, 225, 226, 227, 228, 229, 238, 239, 240, 241, 242, 243, 244, 246, 263, 265, 266, 267, 269,
                270, 271, 272, 278, 279, 280, 281, 282, 283, 284, 285, 289, 290, 291, 292, 293, 294, 295, 296, 298, 299,
                300, 301, 302, 303, 304, 305, 310, 312, 313, 314, 315, 316, 317, 318, 321, 322, 323, 324, 325, 326, 327,
                328, 329, 330, 331, 332, 333, 334, 335, 336, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
                357, 358, 359, 360, 365, 366, 367, 368, 369, 370, 371
            },
            # Special spawn points on highways right before an exit (into town)
            "exit": {22, 67, 216, 268},
            # Special spawn points on highways right before a merge (into another highway lane)
            "merge": {220, 221, 273, 297, 306, 337},
            # Special spawn points on highways right before a junction (with another highway lane)
            "junction": {319, 320}
        },
        "locs": {
            # Location of highway exits (corresponding to exit spawn points)
            "exit": [(36, 0, -4, 75), (35, 0, -4, 280), (42, 0, -4, 30), (48, 0, -4, 60)],
            # Location of highway merges (corresponding to merge spawn points)
            "merge": [(22, 0, -2, 465), (22, 0, -3, 465), (34, 0, 2, 5), (44, 0, 2, 5), (31, 0, 2, 5), (33, 0, 2, 5)],
            # Location of highway junctions (corresponding to junction spawn points)
            "junction": [(23, 0, -2, 425), (23, 0, -3, 425)],
            # Location of highway splits (corresponding to merge and junction spawn points)
            "split": [(135, 0, -1, 55), (136, 0, -1, 55), (782, 1, -2, 35), (1076, 0, 2, 20), (1101, 0, 2, 20),
                      (1191, 1, -2, 35), (1161, 0, 1, 20), (1162, 0, 3, 20)]
        }
    }
}