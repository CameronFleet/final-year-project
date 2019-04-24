import itertools
import sys
sys.path.append("/Users/cameronfleet/Desktop/University/PROJECT/dev")
from policy import make_epsilon_greedy_policy
from stats import Stats
import numpy as np
import os

def save(directory, estimator, stats, name, ep):
    estimator.save(directory + "{}_{}".format(name, ep))
    stats.save_progress(title="{} with {} Episodes".format(name, ep),
                        window_size=20, 
                        path=directory + "PLOT_{}_EPS".format(ep))

def sarsa( env,
            estimator,
            save_dir="",
            job="1",  
            max_episodes = 500, 
            discount_factor = 0.98, 
            epsilon=1, 
            epsilon_decay=0.95, 
            epsilon_min=0.01, 
            early_stopping=200, 
            render=False):

    directory ="weights/{}/{}/".format(save_dir, job)
    os.system("mkdir " + directory)

    stats = Stats(save_dir, job, max_episodes)
   
    for ep in range(max_episodes):

        e = epsilon * epsilon_decay**ep if epsilon * epsilon_decay**ep > epsilon_min else epsilon_min
        policy = make_epsilon_greedy_policy(estimator, e, env.action_space.n)
        
        state = env.reset()
        stats.record(e)
        
        total_t = 0
        action = policy(state)

        for t in itertools.count():
            print("TIME: ",t,end='\r')
            total_t += t

            # Select action using e-greedy policy

            # Step in environment
            next_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            # Stats
            stats.update(ep, reward)
            
            # Update if estimator has update
            next_action = policy(next_state)

            td_target = reward + discount_factor * estimator.v(next_state)[next_action]
            estimator.update(state, action, td_target)

            if done:
                break
                
            state = next_state
            action = next_action

        stop_early = stats.episode_end(early_stopping)

        if ep % 1000 == 0:
            save(directory, estimator, stats, env.name, ep)
        
        if ep % 5 == 0 and stop_early: 
            print(stop_early)
            estimator.save(directory + "BEST_{}".format(int(stop_early)))
    
    save(directory, estimator, stats, env.name, ep)
    return stats

