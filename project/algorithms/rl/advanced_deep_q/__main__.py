import gym
from estimators.fixed_nnestimator import FixedNNEstimator
from estimators.double_nnestimator import DoubleNNEstimator
from advanced_deep_q import q_learning
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

"""
Version 8

estimator = DoubleNNEstimator(env, 
                                learning_rate =0.001, 
                                first_layer_neurons=64, 
                                second_layer_neurons=64)

stats = fixed_q_learning(env, 
                        estimator, 
                        version="8", 
                        max_episodes = 1000, 
                        discount_factor = 0.99, 
                        epsilon=1, 
                        epsilon_decay=0.99, 
                        epsilon_min=0.05, 
                        batch_size= 5, 
                        update_target_network=10000,  
                        learn_every=1,
                        render=False)
"""

"""
Version 9
estimator = DoubleNNEstimator(env, 
                            learning_rate =0.001, 
                            memory_size= 10000,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

stats = fixed_q_learning(env, 
                        estimator, 
                        version="9", 
                        max_episodes = 1000, 
                        discount_factor = 0.99, 
                        epsilon=1, 
                        epsilon_decay=0.99, 
                        epsilon_min=0.05, 
                        batch_size= 5, 
                        update_target_network=10000,  
                        learn_every=1,
                        render=False)
"""

"""
Version 10
estimator = DoubleNNEstimator(  env, 
                                learning_rate =0.001, 
                                memory_size= 100000,
                                first_layer_neurons=64, 
                                second_layer_neurons=64)

stats = q_learning( env, 
                    estimator, 
                    version="10", 
                    max_episodes = 1000, 
                    discount_factor = 0.99, 
                    epsilon=1, 
                    epsilon_decay=0.99, 
                    epsilon_min=0.05, 
                    batch_size= 5, 
                    update_target_network=10000,  
                    learn_every=1,
                    render=False)
stats.plot(100, show=True)
"""

"""
Version 10.2
    max_episodes = 1500, 
    batch_size= 5, 
    learn_every=1,
"""

"""
Version 10.3
    batch_size= 32, 
    learn_every=1,
"""
"""
Version 10.4
    batch_size= 1, 
    learn_every=1,
"""
"""
Version 10.5
estimator = DoubleNNEstimator(env, 
                            learning_rate =0.001, 
                            memory_size= 100000,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

stats = q_learning( env, 
                estimator, 
                version="10.5", 
                max_episodes = 1000, 
                discount_factor = 0.99, 
                epsilon=1, 
                epsilon_decay=0.99, 
                epsilon_min=0.05, 
                batch_size= 5, 
                update_target_network=10000,  
                learn_every=1,
                render=False)
"""

"""
Version 11
estimator = DoubleNNEstimator(env, 
                            learning_rate =0.001, 
                            memory_size= 100000,
                            first_layer_neurons=64, 
                            second_layer_neurons=32)

stats = q_learning( env, 
                estimator, 
                version="10.5", 
                max_episodes = 2000, 
                discount_factor = 0.99, 
                epsilon=1, 
                epsilon_decay=0.99, 
                epsilon_min=0.05, 
                batch_size= 5, 
                update_target_network=10000,  
                learn_every=1,
                render=True)
"""
"""
Version 12
estimator = DoubleNNEstimator(env, 
                        learning_rate =0.001, 
                        memory_size= 100000,
                        first_layer_neurons=64, 
                        second_layer_neurons=64)

stats = q_learning( env, 
                estimator, 
                version="12", 
                max_episodes = 1500, 
                discount_factor = 0.99, 
                epsilon=1, 
                epsilon_decay=0.99, 
                epsilon_min=0.05, 
                batch_size= 5, 
                update_target_network=10000,  
                learn_every=1,
                early_stopping = 200,
                render=True)

stats.plot(100, show=True)
"""

"""
Version 21
def double_train(env, version, episodes= 1500, batch_size= 5):
    estimator = DoubleNNEstimator(env, 
                            learning_rate =0.001, 
                            memory_size= 100000,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

    stats = q_learning( env, 
                    estimator, 
                    version=version, 
                    max_episodes = episodes, 
                    discount_factor = 0.99, 
                    epsilon=1, 
                    epsilon_decay=0.999, 
                    epsilon_min=0.01, 
                    batch_size= batch_size, 
                    update_target_network=10000,  
                    learn_every=1,
                    early_stopping = 200,
                    render=False)

double_train(env, "21", batch_size=64)

"""

"""
def double_train(env, version, episodes= 5000, batch_size= 5):
    estimator = DoubleNNEstimator(env, 
                            learning_rate =0.001, 
                            memory_size= 100000,
                            first_layer_neurons=128, 
                            second_layer_neurons=64)

    stats = q_learning( env, 
                    estimator, 
                    version=version, 
                    max_episodes = episodes, 
                    discount_factor = 0.99, 
                    epsilon=1, 
                    epsilon_decay=0.99, 
                    epsilon_min=0.01, 
                    batch_size= batch_size, 
                    update_target_network=10000,  
                    learn_every=1,
                    early_stopping = 200,
                    render=False)

double_train(env, "22", batch_size=5)
"""

"""
booster_non_moving_2
    estimator = DoubleNNEstimator(env, 
                            learning_rate =0.0001, 
                            memory_size= 100000,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

    stats = q_learning( env, 
                    estimator, 
                    version=version, 
                    max_episodes = episodes, 
                    discount_factor = 0.99, 
                    epsilon=1, 
                    epsilon_decay=0.99, 
                    epsilon_min=0.05, 
                    batch_size= batch_size, 
                    update_target_network=10000,  
                    learn_every=1,
                    early_stopping = 220,
                    render=False)

double_train(env, "_booster_non_moving_2")

"""

env = RocketLander( moving_goal =False,termination_time=1000)
env.name = "RocketLander"
# env = gym.envs.make("CartPole-v1")
# env.name = "CartPole-v1"
# env = gym.envs.make("LunarLander-v2")
# env.name = "LunarLander-v2"


def fixed_train(env, version, episodes=5000, batch_size=5):
    estimator = FixedNNEstimator(env, 
                                learning_rate =0.001, 
                                first_layer_neurons=64, 
                                second_layer_neurons=64)

    stats = q_learning(env, 
                            estimator, 
                            version=version, 
                            max_episodes = episodes, 
                            discount_factor = 0.99, 
                            epsilon=1, 
                            epsilon_decay=0.99, 
                            epsilon_min=0.01, 
                            batch_size= batch_size, 
                            update_target_network=10000,  
                            learn_every=1,
                            early_stopping = 200,
                            render=False)
    stats.plot(10, show=True)

def double_train(env, version, episodes= 5000, batch_size= 5):
    estimator = DoubleNNEstimator(env, 
                            learning_rate =0.00005, 
                            memory_size= 100000,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

    stats = q_learning( env, 
                    estimator, 
                    version=version, 
                    max_episodes = episodes, 
                    discount_factor = 0.99, 
                    epsilon=1, 
                    epsilon_decay=0.995, 
                    epsilon_min=0.01, 
                    batch_size= batch_size, 
                    update_target_network=10000,  
                    learn_every=1,
                    early_stopping = 200,
                    render=True)

double_train(env, "_booster_1")
