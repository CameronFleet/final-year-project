import numpy as np

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    
    # def policy(state):
    #     A = np.ones(nA, dtype=float) * (epsilon / nA)
    #     q_values = estimator.v(state)
    #     best_action = np.argmax(q_values)
    #     A[best_action] += (1.0 - epsilon)
    #     action = np.random.choice(np.arange(nA), p=A)
    #     return action
    def policy(state):
        if np.random.rand() <= epsilon:
            return np.random.choice(np.arange(nA))
        else: 
            return np.argmax(estimator.v(state))
    return policy