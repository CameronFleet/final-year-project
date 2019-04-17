import sys
import gym
import argparse
from estimators.fixed_nnestimator import FixedNNEstimator
from estimators.double_nnestimator import DoubleNNEstimator
from advanced_deep_q import q_learning
from play import play

from project.environment.rocketlander import RocketLander
parser = argparse.ArgumentParser(prog="advanced-q-learning")
parser.add_argument('--save', default="default")
parser.add_argument('--episodes', default=500, type=int)

#env = gym.envs.make("LunarLander-v2")
#env.name = "LunarLander-v2"

env = RocketLander( moving_goal =False,termination_time=1000)
env.name = "RocketLander"

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
                    render=False)

args = parser.parse_args(sys.argv[1:])
double_train(env, args.save, episodes=args.episodes)
