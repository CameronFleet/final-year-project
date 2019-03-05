import numpy as np
import sys
import project.config as config
from project.environment.environment import Env
from project.algorithms.pid_alg import PIDAlg
import time
from project.util.bcolors import bcolors

def test_lander(env, controller, seed=None, render=False):

    actions = []
    step = 0 
    max_step = 4000

    while True: 
        step += 1
        if step % 80 ==0:
            print(bcolors.WARNING + 'STEP: ',round((step/max_step)*100,1) , '%' + bcolors.ENDC, end='\r')
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

        if step > max_step:
            break

    print(bcolors.OKGREEN + 'DONE' + bcolors.ENDC)
    print(bcolors.OKGREEN + 'REWARD: ' + (bcolors.OKGREEN if r > 0 else bcolors.FAIL + str(r) + bcolors.ENDC))

    controller.report(save=True, onlyControl=False)
    return 0

def _record_episode():
    f = open("episodes.log")
    episode_number = len(f.readlines())
    f = open("episodes.log", "a")
    f.write("EPISODE="+ str(episode_number) + " SEED=" + str(env.seed) + "\n")
    return episode_number

if __name__ == '__main__':
    env = Env(True,)
    episode_number = _record_episode()
    alg = PIDAlg(1/config.FPS, env.seed, episode_number)
    test_lander(env, alg, render=True)

