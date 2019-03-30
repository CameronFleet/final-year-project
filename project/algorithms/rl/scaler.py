from sklearn.preprocessing import StandardScaler
import itertools
from project.util.progress import printProgressBar

def scaler(env, sample_size=500):
    observations = []
    for e in range(sample_size):
        printProgressBar(e, sample_size, prefix = 'Scalar Fitting:', suffix = 'Complete', length = 50)
        state = env.reset()
        observations.append(state)
        done = False
        while not done:
                state, _, done, _ =  env.step(env.action_space.sample())
                observations.append(state)
    scaler = StandardScaler()

    scaler.fit(observations)
    return scaler


