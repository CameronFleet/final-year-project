from environment import NoisyBoosterLander, NoisyBoosterLanderContinuous

"""
    When a sensor is 'bad':
    There is a stohastic chance as defined by in the config that
    the given sensor will fail at the begining of each episode. 
    If a sensor fails it does one of two things with a 50/50 chance:
        - Random noisy data
        - Streams the last known value (The value of which is in the initial state)

    accelerometer failure -> bad vx,vy
    gps failure           -> bad x, y
    yaw rate failure      -> bad theta, vtheta
    side thrusters        -> one side thrusters stops functioning 
"""
class BrokenBoosterLander(NoisyBoosterLander):
    good_accelerometer = False
    good_gps = False
    good_rate_sensor = False
    good_side_thrusters = False
    sensors_in_observation = True


class BrokenBoosterLanderContinuous(NoisyBoosterLanderContinuous):
    good_accelerometer = False
    good_gps = False
    good_rate_sensor = False
    good_side_thrusters = False
    sensors_in_observation = True
