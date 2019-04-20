import numpy as np

from pid import Controller, record_episode
from environment import BoosterLander
import environment.config as config
from util import bcolors

env = BoosterLander(continuous=True, time_terminated=False, moving_goal =True)
episode_number = record_episode(env.seed)
controller = Controller(1/config.FPS, env.seed, episode_number)

max_step = 4000
render   = True
report   = False

step = 0 
action = (0,0,0)
cum_reward = 0

while True: 
    step += 1
    if step % 80 ==0:
        print(bcolors.WARNING + 'STEP: ',round((step/max_step)*100,1) , '%' + bcolors.ENDC, end='\r')

    s, r, done, _ = env.step(action)
    cum_reward += r
    x, y, vx, vy,theta, vtheta, l1, l2 = s

    action = controller.action(s, env)

    if l1 or l2:
        action = None

    if render:
        env.render(metrics=True)

    if done: 
        action = None
        break

    if step > max_step:
        break

print(bcolors.OKGREEN + 'DONE' + bcolors.ENDC)
print(bcolors.OKGREEN + 'REWARD: ' + (bcolors.OKGREEN + str(cum_reward) + bcolors.ENDC if r > 0 else bcolors.FAIL + str(r) + bcolors.ENDC))

if report:
    controller.report(save=True, onlyControl=False)
    