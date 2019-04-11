import numpy as np
import sys
import project.config as config
from project.environment.rocketlander import RocketLander
from project.algorithms.pid_alg import PIDAlg
import time
from project.util.bcolors import bcolors

def test_lander(env, controller, seed=None, render=False, report=False):

    action = (0,0,0)
    step = 0 
    max_step = 4000

    while True: 
        step += 1
        if step % 80 ==0:
            print(bcolors.WARNING + 'STEP: ',round((step/max_step)*100,1) , '%' + bcolors.ENDC, end='\r')
        s, r, done, _ = env.step(action)
        x, y, vx, vy,theta, vtheta, l1, l2 = s

        action = controller.go(s, env)

        action = (0.5,0,0)
        if l1 or l2:
            action = None

        if render:
            env.render()

        if done: 
            action = None
            break

        if step > max_step:
            break

    print(bcolors.OKGREEN + 'DONE' + bcolors.ENDC)
    print(bcolors.OKGREEN + 'REWARD: ' + (bcolors.OKGREEN + str(r) + bcolors.ENDC if r > 0 else bcolors.FAIL + str(r) + bcolors.ENDC))

    if report:
        controller.report(save=True, onlyControl=False)
        
    return 0

def _record_episode():
    f = open("episodes.log")
    episode_number = len(f.readlines())
    f = open("episodes.log", "a")
    f.write("EPISODE="+ str(episode_number) + " SEED=" + str(env.seed) + "\n")
    return episode_number

if __name__ == '__main__':
    env = RocketLander(True,time_terminated=False, moving_goal =True)
    episode_number = _record_episode()
    alg = PIDAlg(1/config.FPS, env.seed, episode_number)
    test_lander(env, alg, render=True)

