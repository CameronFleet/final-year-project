from sklearn.preprocessing import StandardScaler
import itertools
from progress.bar import ShadyBar

def scaler(env, sample_size=500):
    observations = []

    bar = ShadyBar('Learning Scaler', max=sample_size)

    for e in range(sample_size):
        bar.next()
        state = env.reset()
        observations.append(state)
        done = False
        while not done:
                state, _, done, _ =  env.step(env.action_space.sample())
                observations.append(state)
    scaler = StandardScaler()

    scaler.fit(observations)
    bar.finish()
    return scaler


