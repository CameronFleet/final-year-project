import sys
import gym
import argparse
from estimators.fixed_nnestimator import FixedNNEstimator
from estimators.double_nnestimator import DoubleNNEstimator
from advanced_deep_q import q_learning
from play import play

from environment.boosterlander import BoosterLander
parser = argparse.ArgumentParser(prog="advanced-q-learning")
parser.add_argument('--save-dir', default="default")
parser.add_argument('--job', default="1")

parser.add_argument('--learning-rate', default=0.00005, type=float)
parser.add_argument('--memory-size', default=100000, type=int)
parser.add_argument('--episodes', default=500, type=int)
parser.add_argument('--batch-size', default=5, type=int)
parser.add_argument('--update-target', default=10000, type=int)

parser.add_argument('-l', default=False, action='store_const', const=True)

args = parser.parse_args(sys.argv[1:])

if args.l:
    env = gym.envs.make("LunarLander-v2")
    env.name = "LunarLander-v2"

if not args.l:
    env = BoosterLander( moving_goal =False,termination_time=1000)
    env.name = "BoosterLander"


estimator = DoubleNNEstimator(env, 
                        learning_rate = args.learning_rate, 
                        memory_size= args.memory_size,
                        first_layer_neurons=64, 
                        second_layer_neurons=64)

stats = q_learning( env, 
                    estimator, 
                    save_dir=args.save_dir,
                    job=args.job, 
                    max_episodes = args.episodes, 
                    discount_factor = 0.99, 
                    epsilon=1, 
                    epsilon_decay=0.995, 
                    epsilon_min=0.01, 
                    batch_size= args.batch_size, 
                    update_target_network=args.update_target,  
                    learn_every=1,
                    early_stopping = 200,
                    render=False)

