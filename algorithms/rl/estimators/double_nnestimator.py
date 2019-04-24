import tensorflow.keras as keras
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import numpy as np
import random
from algorithms.rl.scaler import scaler
import pickle

class DoubleNNEstimator:
    
    def __init__(self, env, learning_rate =0.001, memory_size = 2000, first_layer_neurons=64, second_layer_neurons=64, loaded=False):
        self.model = self._build_model(first_layer_neurons, second_layer_neurons, env.observation_space.shape[0], env.action_space.n, learning_rate)
        self.target_model = self._build_model(first_layer_neurons, second_layer_neurons, env.observation_space.shape[0], env.action_space.n, learning_rate)

        self.memory = deque(maxlen=memory_size)
        if not loaded:
            self.scaler = scaler(env)
        
    def _build_model(self, first_layer_neurons, second_layer_neurons, state_size, action_size, learning_rate):
        model = keras.Sequential()
        model.add(Dense(first_layer_neurons, input_dim=state_size, activation='relu'))
        model.add(Dense(second_layer_neurons, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse',
                           optimizer=Adam(lr=learning_rate))
        return model

    def q(self, state, action):
        return self.v(state)[action]
        
    def v(self, state):
        state = self.scaler.transform(np.array(state).reshape(1,-1))
        return self.model.predict(state)[0]
        
    def remember(self, state, action, reward, next_state, done, next_action=None):
        state = self.scaler.transform(np.array(state).reshape(1,-1))
        self.memory.append((state, action, reward, next_state, done, next_action))
        
    def replay(self, batch_size, discount_rate):
        
        batch_size = batch_size if len(self.memory) > batch_size else len(self.memory)
        batch = random.sample(self.memory, batch_size)
            
        for step in batch: 
            reward, next_state, done, _ = step[2:]
            next_state = self.scaler.transform(np.array(next_state).reshape(1,-1))

            td_target = reward
            if not done:
                best_action = np.argmax(self.model.predict(next_state)[0])
                td_target = (reward + discount_rate * self.target_model.predict(next_state)[0][best_action])

            state, action = step[0:2]
            current = self.model.predict(state)
            current[0][action] = td_target
            self.model.fit(state, current, epochs=1, verbose=0)

        
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def update(self, state, action, target):
        pass

    def save(self, name):
        self.model.save_weights(name+"_weights")
        self.target_model.save_weights(name+"_target_weights")
        pickle.dump(self.scaler, open(name+"_scaler", 'wb'))

    def load(self, name):
        self.model.load_weights(name+"_weights")
        self.target_model.load_weights(name+"_target_weights")
        self.scaler = pickle.load(open(name+"_scaler", 'rb'))