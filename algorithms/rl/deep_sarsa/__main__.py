import gym
from estimators.nnestimator import NNEstimator
from deep_sarsa import sarsa
from play import play

from project.environment.rocketlander import RocketLander

def train(env, estimator):
    stats = sarsa(env, estimator, max_episodes=3000, render=True)
    stats.plot(50)

env = gym.envs.make("LunarLander-v2")
env.name = "LunarLander-v2"
# env = gym.envs.make("CartPole-v1")
# env.name = "CartPole-v1"
# env = RocketLander()
# env.name = "RocketLander"

estimator = NNEstimator(env, 0.001)

train(env, estimator)
