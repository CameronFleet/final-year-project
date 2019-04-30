import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import matplotlib.pyplot as plt
import itertools
import pickle

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from algorithms.rl.scaler import scaler_with_observation

class SGDEstimator:

    def __init__(self, env, lr=0.005, loaded=False):
        self.models = []
        self.env = env

        if not loaded: 
            self._generate_featurizer(env)
            for _ in range(env.action_space.n):
                model = SGDRegressor(learning_rate="constant", eta0=lr)
                model.partial_fit([self.featurize_state(env.reset())], [0]) 
                self.models.append(model)

    def q(self, state, action):
        return self.models[action].predict([self.featurize_state(state)])[0]
    
    def v(self, state):
        return [self.q(state, a) for a in range(len(self.models))]

    def _generate_featurizer(self, env):
        self.scaler, observations = scaler_with_observation(env)
        self.featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                ])
        self.featurizer.fit(self.scaler.transform(observations))

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def update(self, state, action, target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [target])

    def remember(self, state, action, reward, next_state, done, next_action=None):
        pass

    def replay(self, batch_size, discount_rate):
        pass

    def update_target_network(self):
        pass

    def save(self, name):
        pickle.dump(self.scaler, open(name+"_scaler", 'wb'))
        pickle.dump(self.featurizer, open(name+"_featurizer", 'wb'))
        
        for i in range(self.env.action_space.n):
            pickle.dump(self.models[i], open(name+"_{}".format(i), 'wb'))

    def load(self, name):
        self.scaler = pickle.load(open(name+"_scaler", 'rb'))
        self.featurizer = pickle.load(open(name+"_featurizer", 'rb'))

        for i in range(self.env.action_space.n):
            self.models.append(pickle.load(open(name+"_{}".format(i), 'rb')))

