
###########################
#   KERAS IMPORTS
###########################

from keras.models     import Sequential
from keras.layers     import Dense, Activation
from keras.optimizers import SGD   # Stochastic Gradient Descent
from keras.initializers import TruncatedNormal

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

        # ENTRADA:posicion del carro, velocidad del carro, angulo del poste, velocidad de la punta del poste
        self.model.add(Dense(4, input_dim=4, activation='relu'))

        self.model.add(Dense(10, activation='relu'))   # kernel_initializer=TruncatedNormal(mean=1.0,stddev=0.5))

        # Salida = valores Q de las 2 posibles acciones
        self.model.add(Dense(2))

        # Funcion de error: Mean Squared Error; Optimizador: Stochastic Gradient Descent
        self.model.compile(loss='mse', optimizer=SGD(lr=learning_rate), metrics=['accuracy'])

        self.model.summary()

        self.acciones = acciones
        self.gamma = discount_factor
        self.exploration = exploration

    def take_action(self, state):
        if (random.uniform(0, 1) < self.exploration):    # Accion aleatoria
            random.shuffle(self.acciones)
            action = self.acciones[0]
        else:                                            # Mejor accion
            Q = self.model.predict(np.array([state]))
            action = np.argmax(Q)

        return action

    def learn(self,observations):

        batch = []

        contador = 0

        observations.reverse()

        for i in range(len(observations)):

            o = observations.popleft()
            batch.append(o)

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

