import numpy as np
import sys
sys.path.append("C:\\Users\\legac\\Desktop\\Project\\final-year-project\\")
import gym
from estimators.nnestimator import NNEstimator
from estimators.fixed_nnestimator import FixedNNEstimator
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

    estimator = FixedNNEstimator(env,loaded=True)
    estimator.load("weights/FDQN_v5/LunarLander-v2(400-1000)")


    play(env, estimator)

