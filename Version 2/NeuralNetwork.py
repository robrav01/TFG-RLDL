
###########################
#   KERAS IMPORTS
###########################

from keras.models     import Sequential
from keras.layers     import Dense, Activation
from keras.optimizers import SGD   # Stochastic Gradient Descent

import random
import math
import numpy as np

class DQN:
    def __init__(self, acciones, learning_rate, discount_factor, exploration):
        self.model = Sequential()

        # ENTRADA: posicion x + posicion y
        self.model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='random_normal'))

        self.model.add(Dense(20, activation='relu', kernel_initializer='random_normal'))

        # Salida = valores Q de las 4 posibles acciones
        self.model.add(Dense(4))

        # Funcion de error: Mean Squared Error; Optimizador: Stochastic Gradient Descent
        self.model.compile(loss='mse', optimizer=SGD(lr=learning_rate), metrics=['accuracy'])

        self.model.summary()

        self.acciones = acciones
        self.gamma = discount_factor
        self.exploration = exploration

    def take_action(self, state):
        if (random.uniform(0, 1) < self.exploration):    # Accion aleatoria
            random.shuffle(self.acciones)
            for action in self.acciones:
                nextState = self.calculaNextState(state,action)
                if self.validState(nextState):
                    break
        else:                                            # Mejor accion
            Q = self.model.predict(np.array([state]))
            while True:
                action = np.argmax(Q)
                nextState = self.calculaNextState(state, action)

                if not self.validState(nextState):
                    Q[0][action] = -math.inf
                else:
                    break

        return action,nextState

    def learn(self,observations,batch_size):
        minibatch = random.sample(observations, batch_size)

        inputs = np.zeros((batch_size, self.model.input_shape[1]))
        targets = np.zeros((batch_size, len(self.acciones)))

        for i in range(0, batch_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            state_new = minibatch[i][2]
            reward = minibatch[i][3]
            done = minibatch[i][4]

            inputs[i] = state  # ¿np.array(state)?
            targets[i] = self.model.predict(np.array([state]))
            Qnext = self.model.predict(np.array([state_new]))

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(Qnext)

            self.model.fit(inputs, targets, batch_size=batch_size, epochs=5, verbose=False)
        print('learning finished')

    def updateExploration(self,value):
        if self.exploration > 0.005:
            self.exploration += value

    def calculaNextState(self,state,a):
        nextState = state[:]
        if (a == 0):
            nextState[0] -= 1
        elif (a == 1):
            nextState[0] += 1
        elif (a == 2):
            nextState[1] -= 1
        else:
            nextState[1] += 1

        return nextState

    def validState(self,state):
        return (state[0] >= 0 and state[0] <= 7) \
               and (state[1] >= 0 and state[1] <= 7)