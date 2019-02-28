import numpy as np
import sys
import config
from environment.padenv import PadEnv
from algorithms.pid_alg import PIDAlg
from logger import log_episode_begin, log_controller_metrics
import time

def test_lander(env, controller, seed=None, render=False):
    
    actions = []
    env.seed(seed)
    step = 0 

    log_episode_begin(env)

    while True: 
        # Step through world
        step += 1

        # Plant equation
        s, r, done, _ = env.step(actions)

        # State feedback
        x, y, vx, vy,theta, vtheta, alpha, l1, l2 = s

        # Actions selected for next world step
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

    controller.report()
    return 0

if __name__ == '__main__':
    test_lander(PadEnv(True), PIDAlg(1/config.FPS), render=True)



