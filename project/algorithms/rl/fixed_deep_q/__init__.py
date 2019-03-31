import itertools
import sys
sys.path.append("C:\\Users\\legac\\Desktop\\Project\\final-year-project")
from policy import make_epsilon_greedy_policy
from stats import Stats
import numpy as np

import os

LEARN_EVERY = 5

def fixed_q_learning(env, estimator, version="1",  max_episodes = 500, discount_factor = 0.98, epsilon=1, epsilon_decay=0.95, epsilon_min=0.01, batch_size= 30, update_target_network=5000, render=False):

    stats = Stats(max_episodes)
    directory ="weights/FDQN_v{}/".format(version)
    os.system("mkdir " + directory)

    for ep in range(max_episodes):

        e = epsilon * epsilon_decay**ep if epsilon * epsilon_decay**ep > epsilon_min else epsilon_min
        policy = make_epsilon_greedy_policy(estimator, e, env.action_space.n)
        
        state = env.reset()
        stats.record(e)
        
        total_t = 0

        for t in itertools.count():
            print("TIME: ",t,end='\r')
            total_t += t

            # Select action using e-greedy policy
            action = policy(state)

            # Step in environment
            next_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            # Stats
            stats.update(ep, reward)
            
            # Remember experience
            estimator.remember(state, action, reward, next_state, done)
            
            if done:
                break
                
            
            # Experience a replay
            if total_t % LEARN_EVERY == 0:
                estimator.replay(batch_size, discount_factor)

            # Update target network
            if total_t % update_target_network == 0:
                estimator.update_target_network()

            state = next_state



        stats.show()
        if ep % 100 == 0:
            estimator.save(directory + "{}({}-{})".format(env.name, ep, max_episodes))
            stats.save_progress(10, directory + "PLOT_{}_OF_{}".format(ep, max_episodes))

    
    estimator.save(directory + "{}_v{}".format(env.name, version))
    return stats

