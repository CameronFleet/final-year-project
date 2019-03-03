import numpy as np
import sys
import project.config as config
from project.environment.environment import Env
from project.algorithms.pid_alg import PIDAlg
import time

def test_lander(env, controller, seed=None, render=False):

    actions = []
    step = 0 

    while True: 
        step += 1
        s, r, done, _ = env.step(actions)
        x, y, vx, vy,theta, vtheta, alpha, l1, l2 = s

        actions = controller.go(s, env)

        if l1 or l2:
            actions = []

        if render:
            env.render()

        if done: 
            actions = []
            break

        if step > 1000:
            break

    controller.report(save=True, onlyControl=True)
    return 0

def _record_episode():
    f = open("episodes.log")
    episode_number = len(f.readlines())
    f = open("episodes.log", "a")
    f.write("EPISODE="+ str(episode_number) + " SEED=" + str(env.seed) + "\n")
    return episode_number

if __name__ == '__main__':
    env = Env(True)
    episode_number = _record_episode()
    alg = PIDAlg(1/config.FPS, env.seed, episode_number)
    test_lander(env, alg, render=True)



