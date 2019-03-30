import tensorflow.keras as keras
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import numpy as np
import random
from scaler import scaler

class NNEstimator:
    
    def __init__(self, env, learning_rate =0.001, first_layer_neurons=32, second_layer_neurons=64):
        self.model = keras.Sequential()
        self.model.add(Dense(first_layer_neurons, input_dim=env.observation_space.shape[0], activation='relu'))
        self.model.add(Dense(second_layer_neurons, activation='relu'))
        self.model.add(Dense(env.action_space.n, activation='linear'))
        
        self.model.compile(loss='mse',
                           optimizer=Adam(lr=learning_rate))
        
        self.memory = deque(maxlen=2000)
        self.scaler = scaler(env)
        
    def q(self, state, action):
        return self.v(state)[action]
        
    def v(self, state):
        state = self.scaler.transform(np.array(state).reshape(1,-1))
        return self.model.predict(state)[0]
        
    def remember(self, state, action, reward, next_state, done, next_action=None):
        state = self.scaler.transform(np.array(state).reshape(1,-1))
        self.memory.append((state, action, reward, next_state, done, next_action))
        
    def replay(self, batch_size, target_function):
        
        batch_size = batch_size if len(self.memory) > batch_size else len(self.memory)
        batch = random.sample(self.memory, batch_size)
            
        for step in batch: 
            target = target_function(step[2:], self)

            state, action = step[0:2]
            current = self.model.predict(state)
            current[0][action] = target
            self.model.fit(state, current, epochs=1, verbose=0)
                                                
    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)