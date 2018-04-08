
###########################
#   KERAS IMPORTS
###########################

from keras.models     import Sequential
from keras.layers     import Dense, Activation
from keras.optimizers import SGD   # Stochastic Gradient Descent

import random
import math
import numpy as np

# Numero de ejemplos de entrenamiento usados para actualizar los
# pesos cada vez
BATCH_SIZE = 32

# Numero de veces que se expone el modelo a cada conjunto de entrenamiento
EPOCHS = 5

class DQN:
    def __init__(self, acciones, learning_rate, discount_factor, exploration):
        self.model = Sequential()

        # ENTRADA: posicion x + posicion y + moneda obtenida
        self.model.add(Dense(3, input_dim=3, activation='relu', kernel_initializer='random_normal'))

        self.model.add(Dense(10, activation='relu', kernel_initializer='random_normal'))

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

    def learn(self,observations):
        #minibatch = random.sample(observations, batch_size)

        batch = []

        contador = 0

        observations.reverse()

        # Crear el batch con movimientos "utiles"
        for i in range(len(observations)):

            o = observations.pop()

            if o[3] > 0:    # Si la observacion tiene recompensa...
                contador = 30

            if contador > 0: # Si hace menos de 30 movimientos que hubo uno con recompensa
                batch.append(o)
                contador -= 1


        inputs = np.zeros((len(batch), self.model.input_shape[1]))
        targets = np.zeros((len(batch), len(self.acciones)))

        for i in range(0, len(batch)):
            state = batch[i][0]
            action = batch[i][1]
            state_new = batch[i][2]
            reward = batch[i][3]
            done = batch[i][4]

            inputs[i] = state  # ¿np.array(state)?
            targets[i] = self.model.predict(np.array([state]))
            Qnext = self.model.predict(np.array([state_new]))

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(Qnext)

        self.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=False)
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

    def calcularTabla(self):

        # tabla[8][8][2]
        tabla = [[[0 for i in range(2)]for j in range(8)]for k in range(8)]

        for i in range(8):
            for j in range(8):
                for c in range(2):
                    state = [i,j,c]

                    Q = self.model.predict(np.array([state]))
                    while True:
                        action = np.argmax(Q)
                        nextState = self.calculaNextState(state, action)

                        if not self.validState(nextState):
                            Q[0][action] = -math.inf
                        else:
                            break

                    tabla[i][j][c] = action

        return tabla
