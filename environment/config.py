FPS = 60
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 40
SIDE_ENGINE_POWER = 5

# (px)
VIEWPORT_W = 1300
VIEWPORT_H = 800

# (m)
WORLD_SCALAR = 0.5
WORLD_W = VIEWPORT_W * WORLD_SCALAR
WORLD_H = VIEWPORT_H * WORLD_SCALAR

# (m)
GOAL_H = 5
GOAL_W = 100

# (m)
LANDER_POLY = [
    (-1.5, +40), (-1.5, 0), 
    (+1.5, +40), (+1.5, 0),
]
# (m)
LEG_AWAY = 4
LEG_DOWN = 2
LEG_W, LEG_H = 0.5, 4
LEG_SPRING_TORQUE = 1000

# (m/s)
START_VELOCITY = (0,0)

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 5
