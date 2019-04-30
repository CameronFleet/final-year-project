import sys
import gym
import argparse

from environment import BoosterLander, NoisyBoosterLander, BrokenBoosterLander
from algorithms.rl.estimators import SGDEstimator, DoubleNNEstimator, FixedNNEstimator, NNEstimator
from algorithms.rl.q_learning import q_learning

parser = argparse.ArgumentParser(prog="q-learning")
parser.add_argument('--save-dir', default="default", help='Save directory for the experiment')
parser.add_argument('--job', default="1", help='Unique identifier for this job')

parser.add_argument('--learning-rate', default=0.00005, type=float, help='Learning rate for the estimator')
parser.add_argument('--memory-size', default=100000, type=int, help='DDQN ONLY. Memory size of the memory used in experience replay')
parser.add_argument('--episodes', default=500, type=int, help='Number of episodes the job should run until')
parser.add_argument('--batch-size', default=5, type=int, help='DDQN ONLY. Size of the batch of experiences that is used in experience replay')
parser.add_argument('--update-target', default=10000, type=int, help='DDQN ONLY. The rate at which the target network should be updated')

parser.add_argument('-c', default=False, action='store_const', const=True, help='Environment is CartPole-v0')
parser.add_argument('-l', default=False, action='store_const', const=True, help='Environment is LunarLander-v2')
parser.add_argument('-n', default=False, action='store_const', const=True, help='Environment is NoisyBoosterLander')
parser.add_argument('-b', default=False, action='store_const', const=True, help='Environment is BrokenBoosterLander')

parser.add_argument('-e', default="DDQN", choices=['DDQN','DQN','NN','SGD'], help='The estimator used in the training')

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

print("ENV: {}  ESTIMATOR: {}  ALGORITHM: Q-LEARNING".format(env.name, args.e))

# DEFINE ESTIMATOR
if args.e == "DDQN":
    estimator = DoubleNNEstimator(env, 
                            learning_rate = args.learning_rate, 
                            memory_size= args.memory_size,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)
elif args.e == "DQN":
    estimator = FixedNNEstimator(env, 
                            learning_rate = args.learning_rate, 
                            memory_size= args.memory_size,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)

elif args.e == "NN":
    estimator = NNEstimator(env, 
                            learning_rate = args.learning_rate, 
                            memory_size= args.memory_size,
                            first_layer_neurons=64, 
                            second_layer_neurons=64)
elif args.e == "SGD":
    estimator = SGDEstimator(env,
                        lr=args.learning_rate)
else: 
    print("INVALID ESTIMATOR [{}]".format(args.e))
    quit()


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
                    early_stopping = 100,
                    render=False)

