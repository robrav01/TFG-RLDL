import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from NeuralNetwork import DQN

from collections import deque

from GUI import Maze

################################
#   GLOBAL CONSTANTS
################################

# Numero de movimientos observados antes de aprender
OBSERVACIONES = 300

# Numero de ejemplos de entrenamiento usados para actualizar los
# pesos cada vez
BATCH_SIZE = 50

# Numero de veces que se expone el modelo a cada conjunto de entrenamiento
EPOCHS = 5

exploration = 0.50
alpha = 0.1
gamma = 0.9

init_state = [0,0]
coin_pos   = [2,1]
end_state  = [7,7]

acciones = [0, 1, 2, 3]

def execute(times):
    for t in range(times):
        print(t)

        # INICIALIZACION
        state     = init_state
        numMov    = 0               # Numero de movimientos
        coinObtAt = 0               # Numero de movimientos necesitados para coger la moneda
        coinObt   = False           # Si se ha cogido o no la moneda
        done      = False           # Si se ha terminado (llegado al estado final)
        observaciones = deque()

        for o in range(OBSERVACIONES):

            action, next = DQN.take_action(state)
            GUI.visualize(state, next, o, t);

            if next == end_state:
                reward = 5 / (numMov + 1)
                done = True
            elif next == coin_pos and not coinObt:
                coinObt = True
                reward = 15 / (numMov + 1)
                coinObtAt = numMov + 1
            else:
                reward = 0

            observaciones.append((state, action, next, reward, done))

            state = next[:]
            numMov += 1

            if done:
                GUI.visualize(state,init_state,o,t)
                print('Finished in %i movements, coin obtained in %i' % (numMov, coinObtAt))
                state     = init_state
                numMov    = 0
                coinObtAt = 0
                coinObt   = False
                done      = False

                DQN.updateExploration(-0.005)

        print('Observation finished')
        GUI.visualize(state,init_state,o,t)

        DQN.learn(observaciones,BATCH_SIZE)


if __name__ == '__main__':
        DQN = DQN(acciones,alpha,gamma,exploration)
        GUI = Maze(8,8)
        execute(100)