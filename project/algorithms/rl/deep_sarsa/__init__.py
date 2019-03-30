import itertools
import sys
sys.path.append("/Users/cameronfleet/Desktop/University/PROJECT/dev/")
from policy import make_epsilon_greedy_policy
from stats import Stats

def sarsa_target(discount_rate):
    def target_fn(step, estimator):

        reward, next_state, done, next_action = step
        target = reward
        if not done:
            target = (reward + discount_rate * estimator.q(next_state, next_action))

        return target

    return target_fn

def sarsa(env, estimator, max_episodes = 500, discount_factor = 0.99, epsilon=1, epsilon_decay=0.98, epsilon_min=0.05, render=False):

    stats = Stats(max_episodes)

    for ep in range(max_episodes):

        e = epsilon * epsilon_decay**ep if epsilon * epsilon_decay**ep > epsilon_min else epsilon_min
        policy = make_epsilon_greedy_policy(estimator, e, env.action_space.n)
        
        state = env.reset()
        stats.record(e)
        
        # Select an initial action
        action = policy(state)
        
        for t in itertools.count():
            
            # Step in environment
            print("TIME: ",t,end='\r')
            next_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            # Stats
            stats.update(ep, reward)
            # Update based on the TD target + highest next value
            next_action = policy(next_state)
            
            estimator.remember(state, action, reward, next_state, done, next_action)
            
            if done:
                break
                
            state = next_state
            action = next_action
            estimator.replay(50, sarsa_target(discount_factor))

        
        stats.show()

        if ep % 200 == 0:
            name = "weights/DSN:{},eps={},gamma={},epsilon={},epsilon_decay={},epsilon_min={}".format(
                env.name,
                ep,
                discount_factor,
                epsilon,
                epsilon_decay,
                epsilon_min
            )
            estimator.save(name)

    name = "weights/DSN:{},eps={},gamma={},epsilon={},epsilon_decay={},epsilon_min={}".format(
                env.name,
                max_episodes,
                discount_factor,
                epsilon,
                epsilon_decay,
                epsilon_min
            )
    estimator.save(name)
    return stats

