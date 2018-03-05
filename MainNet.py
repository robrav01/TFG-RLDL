from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# https://repl.it/NDgQ/4
import random
import math
# KERAS IMPORTS

from keras.datasets   import mnist
from keras.models     import Sequential
from keras.layers     import Dense, Activation
from keras.optimizers import SGD   # Stochastic Gradient Descent
from keras.utils      import to_categorical

from collections import deque

import numpy as np

# GLOBALS
OBSERVACIONES = 500

BATCH_SIZE = 50

# Posicion de la moneda
MONEDA = [3,4]

exploration = 0.50
alpha = 0.1
gamma = 0.9  # 0.05

####### NEURAL NETWORK ########
model = Sequential();

# 8 bits codifican posicion x + 8 bits codifican posicion y = 16
model.add(Dense(10,input_dim=2,activation='relu',kernel_initializer='random_uniform'))

model.add(Dense(20,activation='relu',kernel_initializer='random_uniform'))

# Salida = 4 valores Q de las 4 posibles acciones
model.add(Dense(4))

# Funcion de error: Mean Squared Error; Optimizador: Stochastic Gradient Descent
model.compile(loss='mse',optimizer=SGD(lr=alpha),metrics=['accuracy'])

model.summary()

def validState(state):
    return (state[0] >= 0 and state[0] <= 7) \
           and (state[1] >= 0 and state[1] <= 7)

def calculaNextState(state,a):
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

def onehot(state):
    return (to_categorical(state,8))

def execute():
    global exploration
    state = [0, 0]
    nextState = [0, 0]
    numMov = 0
    coinObtAt = 0

    D = deque()
    acciones = [0, 1, 2, 3]
    moneda_cogida = False
    done = False

    # 1 -> OBSERVAR
    for t in range(OBSERVACIONES):
        if (random.uniform(0, 1) < exploration):    # Accion aleatoria
            random.shuffle(acciones)
            for action in acciones:
                nextState = calculaNextState(state,action)
                if validState(nextState):
                    break
        else:                                       # Mejor accion
            Q = model.predict(np.array([state]))
            #print(Q)
            while True:
                action = np.argmax(Q)
                nextState = calculaNextState(state, action)

                if not validState(nextState):
                    Q[0][action] = -math.inf
                else:
                    break

        #print('Estado', state)
        #print('Accion', action)
        #print('Estado', nextState)

        # Give reward
        if nextState == [7, 7]:
            reward = 5 / (numMov + 1)
            done = True
        elif nextState == MONEDA and not moneda_cogida:
            reward = 5 / (numMov + 1)
            coinObtAt = numMov + 1
            moneda_cogida = True
        else:
            reward = 0

        D.append((state,action,nextState,reward,done))

        '''
        maxQnext = -math.inf
        for i in range(4):
            Qnext = model.predict(np.array([[nextState[0],nextState[1],nextState[2],i]]))
            if Qnext[0][0] > maxQnext:
                maxQnext = Qnext[0][0]

        t = (1 - alpha) * model.predict(np.array([[state[0],state[1],state[2],action]])) \
                 + alpha * (reward + gamma * maxQnext)
        #print(t[0])
        '''

        state = nextState[:]
        numMov += 1

        if done:
            print('Finished in %i movements, coin obtained in %i' % (numMov, coinObtAt))
            state = [0, 0]
            nextState = [0, 0]
            numMov = 0
            coinObtAt = 0
            moneda_cogida = False
            done = False
            if (exploration > 0.05):
                exploration -= 0.0001

    print('Observation finished')

    # 2 -> APRENDER
    minibatch = random.sample(D,BATCH_SIZE)

    inputs = np.zeros((BATCH_SIZE,len(state)))
    targets = np.zeros((BATCH_SIZE,len(acciones)))

    for i in range(0,BATCH_SIZE):
        state = minibatch[i][0]
        action = minibatch[i][1]
        state_new = minibatch[i][2]
        reward = minibatch[i][3]
        done = minibatch[i][4]

        inputs[i] = state       # ¿np.array(state)?
        targets[i] = model.predict(np.array([state]))
        Qnext = model.predict(np.array([state_new]))

        # COMO TARGET NO ES NECESARIO MANTENER CONOCIMIENTO ACTUAL YA QUE ESO LO HACE LA
        # PROPIA RED NEURONAL AL ENTRENARSE, POR TANTO LE PASAMOS COMO OBJETIVO SIMPLEMENTE
        # EL VALOR ESPERADO
        # (AHORA NO ES UNA ASIGNACIÓN QUE PISARIA LO APRENDIDO, SINO QUE ES UN VALOR OBJETIVO
        # PARA AJUSTAR LOS PESOS)
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Qnext)

        model.fit(inputs,targets,batch_size=BATCH_SIZE,epochs=5,verbose=False)
    print('learning finished')

    #states = [[x,y] for x in range(8) for y in range(8)]

    #print('Valores Q', model.predict(np.array(states)))

def execute100():
    for i in range(100):
        print(i)
        execute()


def execute1000():
    for i in range(1000):
        print(i)
        execute()


if __name__ == '__main__':
    execute1000()