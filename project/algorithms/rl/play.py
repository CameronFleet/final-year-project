import numpy as np
import sys
sys.path.append("/Users/cameronfleet/Desktop/University/PROJECT/dev/")
import gym
from estimators.nnestimator import NNEstimator
from project.environment.rocketlander import RocketLander


def play(env, estimator):

    state = env.reset()
    action =  np.argmax(estimator.v(state))
    done = False
    total_reward = 0

    while not done:
        state, reward, done, _ = env.step(action)
        env.render()

        total_reward += reward
        print(estimator.v(state))
        action = np.argmax(estimator.v(state))

    print("Finished with reward {}".format(total_reward))

if __name__ == '__main__':
    # env = gym.envs.make("CartPole-v1")
    # env.name = "CartPole-v1"
    # env = RocketLander()
    # env.name = "RocketLander"
    env = gym.envs.make("LunarLander-v2")
    env.name = "LunarLander-v2"

    estimator = NNEstimator(env)
    estimator.load("weights/DSN:LunarLander-v2,eps=2800,gamma=0.99,epsilon=1,epsilon_decay=0.98,epsilon_min=0.05")

    play(env, estimator)

