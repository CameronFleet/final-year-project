from lunarlander import (LunarLander, LunarLanderContinuous)
import numpy as np
import sys
# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# Too see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.


def heuristic(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0] * 0.5 + s[
        2] * 1.0  # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ > 0.4: angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])  # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    # print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
    # print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            #print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            #print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            continue

        print(steps)
        if steps > 100:
            env.render(close=True)
            break
            quit()

        steps += 1
        if done: break
    return total_reward

def test_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        s = env.step(2)

        if render:
            still_open = env.render()
            if still_open == False: break

        steps += 1
    return total_reward


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        test_lander(LunarLander(), render=True)
    else:
        demo_heuristic_lander(LunarLander(), render=True)

