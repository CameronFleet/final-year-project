from environment import BoosterLander, BoosterLanderContinuous
import environment.config as config

class NoisyBoosterLander(BoosterLander):   
    initial_random = 15000.0

class NoisyBoosterLanderContinuous(BoosterLanderContinuous):   
    initial_random = 15000.0