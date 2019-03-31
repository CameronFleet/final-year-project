import gym
from estimators.fixed_nnestimator import FixedNNEstimator
from fixed_deep_q import fixed_q_learning
from play import play

from project.environment.rocketlander import RocketLander



"""
Version 2
set:
version="2", max_episodes = 2000, discount_factor = 0.99, epsilon=1, epsilon_decay=0.97, epsilon_min=0.1, update_target_network=5000, render=True
learning_rate =0.001, first_layer_neurons=64, second_layer_neurons=64

defaults:
max_episodes = 500, discount_factor = 0.98, epsilon=1, epsilon_decay=0.95, epsilon_min=0.01, batch_size= 30, update_target_network=5000, render=False
"""


"""
Version 3

notes: experience replay is every episode
learning_rate =0.001, 
first_layer_neurons=64, 
second_layer_neurons=64,
max_episodes = 1000, 
discount_factor = 0.99, 
epsilon=1, 
epsilon_decay=0.99, 
epsilon_min=0.05, 
batch_size= 128, 
update_target_network=10000

"""

"""
Version 5

notes: experience replay is every timestep
learning_rate =0.001, 
first_layer_neurons=64, 
second_layer_neurons=64
max_episodes = 1000, 
discount_factor = 0.99, 
epsilon=1, 
epsilon_decay=0.99, 
epsilon_min=0.05, 
batch_size= 5, 
update_target_network=10000,  
render=True

"""

"""
Version 7

estimator = FixedNNEstimator(env, 
                            learning_rate =0.001, 
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

stats = fixed_q_learning(env, 
                        estimator, 
                        version="7", 
                        max_episodes = 1000, 
                        discount_factor = 0.99, 
                        epsilon=1, 
                        epsilon_decay=0.99, 
                        epsilon_min=0.1, 
                        batch_size= 64, 
                        update_target_network=10000,  
                        learn_every=5
                        render=False)
"""

env = RocketLander()
env.name = "RocketLander"
# env = gym.envs.make("LunarLander-v2")
# env.name = "LunarLander-v2"


# env = gym.envs.make("CartPole-v1")
# env.name = "CartPole-v1"

estimator = FixedNNEstimator(env, 
                            learning_rate =0.001, 
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

stats = fixed_q_learning(env, 
                        estimator, 
                        version="7", 
                        max_episodes = 1000, 
                        discount_factor = 0.99, 
                        epsilon=1, 
                        epsilon_decay=0.99, 
                        epsilon_min=0.01, 
                        batch_size= 64, 
                        update_target_network=10000,  
                        learn_every=5,
                        render=False)
stats.plot(10, show=True)


