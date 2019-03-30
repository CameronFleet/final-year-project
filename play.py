import pickle
import numpy as np
import gym

from q_learning import SGDEstimator
from project.environment.rocketlander import RocketLander

class BaseEstimator:
    def __init__(self, action_space):
        self.action_space = action_space

    def v(self, s):
        return self.action_space.sample()

def play(env, estimator):
    
    s = env.reset()
    action = np.argmax(estimator.v(s))
    done = False

    total_reward = 0
    # for each timestep until terminal timestep T
    while not done: 
        s, r, done, _ = env.step(action)

        # as a function of the value function q
        action = np.argmax(estimator.v(s))

        total_reward += r 

        env.render()

        if done:
            env.close()

    print(total_reward)

if __name__ == '__main__':
    env       = RocketLander()

    with open('estimator.SGD.5.6.SARSA.pkl', 'rb') as  f:
        estimator = pickle.load(f)
    # estimator = BaseEstimator(env.action_space)

    play(env, estimator)


    