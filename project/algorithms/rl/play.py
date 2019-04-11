import numpy as np
import sys
sys.path.append("C:\\Users\\legac\\Desktop\\Project\\final-year-project\\")
import gym
from estimators.nnestimator import NNEstimator
from estimators.fixed_nnestimator import FixedNNEstimator
from estimators.double_nnestimator import DoubleNNEstimator
from project.environment.rocketlander import RocketLander
import time

def play(env, estimator, debug=False):

    state = env.reset()
    action =  np.argmax(estimator.v(state))
    done = False
    total_reward = 0

    while not done:
        state, reward, done, _ = env.step(action)
        env.render()

        if debug:
            print(state[0])

        total_reward += reward
        if debug:
            print(estimator.v(state))
        action = np.argmax(estimator.v(state))

    return total_reward

def test(env, estimator, n= 100):

    total = 0

    for i in range(n):
        reward = play(env, estimator)
        total += reward
        print("Iteration {} , Reward {}".format(i, reward))

    return total/n
if __name__ == '__main__':
    # env = gym.envs.make("CartPole-v1")
    # env.name = "CartPole-v1"
    env = RocketLander(time_terminated=True, moving_goal=False)
    env.name = "RocketLander"
    # env = gym.envs.make("LunarLander-v2")
    # env.name = "LunarLander-v2"

    estimator = DoubleNNEstimator(env,loaded=True)
    estimator.load("weights/v_booster_1/RocketLander_BEST_PERFORMING_211")


    print(test(env, estimator))
    # play(env, estimator)

