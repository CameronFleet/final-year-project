import numpy as np
import sys
from environment import *
from algorithms import *
from environment.physics import fuel_usage
import argparse
import itertools
import os
"""
TO RUN
 python evaluation.py -env-b -dqn /Users/cameronfleet/Desktop/eval/bl_broken_better_reward_2/bbl_acc0_gps0_rate0_thrust0.5/7/BEST_277  --save side_booster_fail
#  python evaluation.py -env-n -dqn /Users/cameronfleet/Desktop/eval/best/noisy_and_nominal/BEST_248
"""
parser = argparse.ArgumentParser(prog="evaluation")
parser.add_argument('--save-dir', default="default")

parser.add_argument('-r', default=True, action='store_const', const=False, help="Do not render")
parser.add_argument('-m', default=False, action='store_const', const=True, help="Display metrics on render")

parser.add_argument('-env-n', default=False, action='store_const', const=True, help="Runs on noisy environment")
parser.add_argument('-env-b', default=False, action='store_const', const=True, help="Runs on broken environment")

parser.add_argument('-dqn', default=False, help="Path to DQN saved estimator")

parser.add_argument('--tests', default=150, help="Number of tests to run when evaluating", type=int)
parser.add_argument('--save', default=False, help="Path to save results")


args = parser.parse_args(sys.argv[1:])


def play(env, action_fn):

    observation = env.reset()
    action = action_fn(observation)

    done = False
    total_reward = 0
    total_Fs = 0
    total_Ft = 0

    for t in itertools.count():
        observation, reward, done, performance_metrics = env.step(action)

        if args.r:
            env.render(metrics=args.m)

        total_reward += reward
        total_Ft += performance_metrics['Ft'] 
        total_Fs += performance_metrics['Fs'] 

        action = action_fn(observation)

        if done:
            landed = performance_metrics['landed'] 
            impulse = performance_metrics['impulse']
            fuel_mass = fuel_usage(t/env.T, env.booster.mfr, total_Ft/t)
            return {"Total Reward": total_reward, 
                    "Cold Gas Utilisation":abs(total_Fs/t), 
                    "Fuel Expenditure":fuel_mass, 
                    "Impact impulse":impulse,
                    "Booster Landed":int(landed)}


def evaluate(env, action_fn, controller=None):

    performance_metrics = []

    for i in range(args.tests):
        performance = play(env, action_fn)
        print(performance)
        performance_metrics.append(performance)
        if controller:
            controller.reset()

    avg_metrics = {}
    fails = 0

    for key in performance_metrics[0]:
        metric = [ metric[key] for metric in performance_metrics]
        cleaned_metric = [val for val in metric if val is not None]
        fails += len(metric) - len(cleaned_metric)
        avg_metrics[key] = sum(cleaned_metric) / len(performance_metrics)

    avg_metrics["Fails"] = fails
    return avg_metrics

if __name__ == '__main__':

    if args.env_n:
        env = NoisyBoosterLander(termination_time=1500) if args.dqn else NoisyBoosterLanderContinuous(termination_time=1500)
        env.name = "NoisyBoosterLander"
   
    elif args.env_b:
        env = BrokenBoosterLander(termination_time=1500) if args.dqn else BrokenBoosterLanderContinuous(termination_time=1500)
        env.name = "BrokenBoosterLander"
    else: 
        env = BoosterLander(termination_time=1500) if args.dqn else BoosterLanderContinuous(termination_time=1500)
        env.name = "BoosterLander"

    controller = None

    print(args.save)

    if args.dqn:
        estimator = DDQNEstimator(env,loaded=True)
        alg = "DDQN"
        estimator.load(args.dqn)
        action_fn = lambda state: np.argmax(estimator.v(state))
    else: 
        controller = PIDController(env)
        alg = "PID"
        action_fn = lambda observation: controller.action(observation)

    avg_metrics = evaluate(env, action_fn, controller)

    if args.save:
        file = "evaluation/{}.txt".format(args.save)
        os.system("touch {}".format(file))
        f = open(file, "a")
        f.write("=== EVAL FOR {} TESTS OF {} USING {} ===\n".format(args.tests, env.name, alg))
        for metric in avg_metrics:
            f.write("{} = {}\n".format(metric, avg_metrics[metric]))
        f.write("========================================\n")
        f.close()
        




