###########################
#   IMPORTS
###########################

import random
import numpy as np

from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

from collections import deque

MEMORY_SIZE = 5000
EPOCHS = 5

def one_hot(x):
    l = [0]*500
    l[x] = 1
    return l

class DQN:
    def __init__(self, state_space, action_space, learning_rate, discount_factor, exploration):

        ### Enviroment spaces
        self.state_space = state_space
        self.action_space = action_space

        ### Agent's memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        ### Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration

        self.epsilon_min = 0.01

        ### Neural Network's architecture
        self.model = Sequential()

        # input layer - 2x hidden layers - output layer
        self.model.add(Dense(units = 50, input_dim=state_space, activation='relu'))
        self.model.add(Dense(units = 15, activation='relu'))
        self.model.add(Dense(units = action_space, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)

        # Define the loss function and the optimizer used to minimize it
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def act(self,state):
        if (random.uniform(0, 1) < self.epsilon):   # Random action
            action = random.randrange(self.action_space)
        else:
            # Predict is meant to get a batch, where rows are samples
            # and columns are samples' dimensions. Then, when you give it a
            # single sample you have to cast it to a 2D array (batch of samples)
            # and get the first (unique) prediction
            Q = self.model.predict(np.array([one_hot(state)]))[0] # Best action
            action = np.argmax(Q)

        return action

    def add_to_memory(self, observation):
        self.memory.append(observation)

    def get_minibatch(self, batch_size):
        indices = np.random.choice(np.arange(len(self.memory)),size=batch_size, replace=False)
        return [self.memory[i] for i in indices]

    def replay(self):

        batch = list(self.memory)
        batch.reverse()

        inputs = np.zeros((len(batch),self.state_space))
        targets = np.zeros((len(batch),self.action_space))

        i = 0

        for state, action, next_state, reward, done in batch:

            inputs[i] = one_hot(state)
            targets[i] = self.model.predict(np.array([one_hot(state)]))[0]

            if done:
                targets[i, action] = reward
            else:
                # Bellman's equation
                targets[i, action] = reward + self.gamma * \
                                     np.max(self.model.predict(np.array([one_hot(next_state)]))[0])

            i = i + 1

        self.model.fit(inputs, targets, batch_size=len(batch), epochs=EPOCHS, verbose=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.9994











