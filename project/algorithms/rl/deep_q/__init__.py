import itertools
import sys
sys.path.append("/Users/cameronfleet/Desktop/University/PROJECT/dev/")
from policy import make_epsilon_greedy_policy
from stats import Stats
import numpy as np

def q_target(discount_rate):
    def target_fn(step, estimator):

        reward, next_state, done, _ = step
        target = reward
        if not done:
            target = (reward + discount_rate * np.amax(estimator.v(next_state)))

        return target

    return target_fn

def q_learning(env, estimator, max_episodes = 500, discount_factor = 0.98, epsilon=1, epsilon_decay=0.95, epsilon_min=0.01, render=False):

    stats = Stats(max_episodes)

    for ep in range(max_episodes):

        e = epsilon * epsilon_decay**ep if epsilon * epsilon_decay**ep > epsilon_min else epsilon_min
        policy = make_epsilon_greedy_policy(estimator, e, env.action_space.n)
        
        state = env.reset()
        stats.record(e)
        
        for t in itertools.count():
            print("TIME: ",t,end='\r')
            
            action = policy(state)
            # Step in environment
            next_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            # Stats
            stats.update(ep, reward)
            
            estimator.remember(state, action, reward, next_state, done)
            
            if done:
                break
                
            state = next_state
            estimator.replay(30, q_target(discount_factor))

        stats.show()
    
    name = "DQN:{},eps={},gamma={},epsilon={},epsilon_decay={},epsilon_min={}".format(
        env.name,
        max_episodes,
        discount_factor,
        epsilon,
        epsilon_decay,
        epsilon_min
    )
    estimator.save(name)
    return stats

