import sys
import gym
import argparse

from environment import BoosterLander, NoisyBoosterLander, BrokenBoosterLander
from algorithms.rl.estimators import SGDEstimator, DoubleNNEstimator, FixedNNEstimator, NNEstimator
from algorithms.rl.sarsa import sarsa

parser = argparse.ArgumentParser(prog="advanced-q-learning")
parser.add_argument('--save-dir', default="default")
parser.add_argument('--job', default="1")

parser.add_argument('--learning-rate', default=0.00005, type=float)
parser.add_argument('--episodes', default=500, type=int)

parser.add_argument('-c', default=False, action='store_const', const=True)
parser.add_argument('-l', default=False, action='store_const', const=True)
parser.add_argument('-n', default=False, action='store_const', const=True)
parser.add_argument('-b', default=False, action='store_const', const=True)

args = parser.parse_args(sys.argv[1:])

# DEFINE ENV
if args.c:
    env = gym.envs.make("CartPole-v0")
    env.name = "CartPole-v0"
elif args.l:
    env = gym.envs.make("LunarLander-v2")
    env.name = "LunarLander-v2"
elif args.n:
    env = NoisyBoosterLander( moving_goal =False,termination_time=1000)
    env.name = "BoosterLander"
elif args.b:
    env = BrokenBoosterLander( moving_goal =False,termination_time=1000)
    env.name = "BoosterLander"
else:
    env = BoosterLander( moving_goal =False,termination_time=1000)
    env.name = "BoosterLander"

print("ENV: {}  ESTIMATOR: SGD  ALGORITHM: SARSA".format(env.name))

estimator = SGDEstimator(env,
                    lr=args.learning_rate)

stats = sarsa(  env, 
                estimator, 
                save_dir=args.save_dir,
                job=args.job, 
                max_episodes = args.episodes, 
                discount_factor = 0.99, 
                epsilon=1, 
                epsilon_decay=0.995, 
                epsilon_min=0.01, 
                early_stopping = 100,
                render=False)

