import gym
from estimators.nnestimator import NNEstimator
from deep_q import q_learning
from play import play

from project.environment.boosterlander import BoosterLander

# env = gym.envs.make("CartPole-v1")
# env.name = "CartPole-v1"
# env = gym.envs.make("LunarLander-v2")
# env.name = "LunarLander-v2"

env = BoosterLander()
env.name = "BoosterLander"

estimator = NNEstimator(env, 0.001)

def train(env, estimator):
    stats = q_learning(env, estimator, max_episodes=200, render=True, discount_factor = 0.99, epsilon=0.9, epsilon_decay=0.9, epsilon_min=0.005)
    stats.plot(10)

train(env, estimator)
