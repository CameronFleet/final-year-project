import itertools
import sys
sys.path.append("C:\\Users\\legac\\Desktop\\Project\\final-year-project")
from policy import make_epsilon_greedy_policy
from stats import Stats
import numpy as np
import os

def save(directory, estimator, stats, name, version, ep):
    estimator.save(directory + "{}_{}".format(name, ep))
    stats.save_progress(title="{} with {} Episodes with v{} Hyperparameters".format(name, ep, version),
                                window_size=20, 
                                path=directory + "PLOT_{}_EPS".format(ep))



def q_learning( env,
                estimator, version="1",  
                max_episodes = 500, 
                discount_factor = 0.98, 
                epsilon=1, 
                epsilon_decay=0.95, 
                epsilon_min=0.01, 
                batch_size= 30, 
                update_target_network=5000, 
                learn_every=1,
                early_stopping=200, 
                render=False):

    stats = Stats(max_episodes)
    directory ="weights/v{}/".format(version)
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
            if total_t % learn_every == 0:
                estimator.replay(batch_size, discount_factor)

            # Update target network
            if total_t % update_target_network == 0:
                estimator.update_target_network()

            state = next_state

        stop_early = stats.episode_end(early_stopping)

        if ep % 100 == 0:
            save(directory, estimator, stats, env.name, version, ep)
        
        if ep % 5 == 0 and stop_early: 
            print(stop_early)
            save(directory, estimator, stats, env.name, version, "BEST_PERFORMING_{}".format(int(stop_early)))
            
    
    save(directory, estimator, stats, env.name, version, ep)
    return stats

