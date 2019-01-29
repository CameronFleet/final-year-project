FPS = 60
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 100
SIDE_ENGINE_POWER = 0

# (px)
VIEWPORT_W = 1300
VIEWPORT_H = 800

# (m)
WORLD_SCALAR = 3
WORLD_W = VIEWPORT_W * WORLD_SCALAR
WORLD_H = VIEWPORT_H * WORLD_SCALAR

# (m)
GOAL_H = 30
GOAL_W = 500

# (m)
LANDER_POLY = [
    (-10, +190), (-10, 0), 
    (+10, +190), (+10, 0),
]
# (m)
LEG_AWAY = 20
LEG_DOWN = 15
LEG_W, LEG_H = 4, 24
LEG_SPRING_TORQUE = 40

# (m/s)
START_VELOCITY = (0,-100)

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0
