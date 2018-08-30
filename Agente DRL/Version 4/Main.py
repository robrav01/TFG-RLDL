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

exploration = 1.0
alpha = 0.05
gamma = 0.9

init_state = [0,0,0]
coin_pos   = [3,4]
end_state  = [7,7,1]

# 0 -> Arriba ; 1 -> Abajo ; 2 -> Izquierda ; 3 -> Derecha
acciones = [0, 1, 2, 3]

# Meter moneda en el estado #
# Obtener lista de N movimientos. Buscar movimientos con recompensa y aprender con los k anteriores
# Meter ejemplos de entrenamiento en el batch en orden inverso
# Codificar estado en one-hot

# Interfaz (mapa de calor, ...)


# Si no obtiene recompensa en los 500 movimientos, descartar esa "ronda"

def execute(times):
    for t in range(times):
        print(t)

        # INICIALIZACION
        state     = init_state
        numMov    = 0               # Numero de movimientos
        coinObtAt = 0               # Numero de movimientos necesitados para coger la moneda
        coinObt   = False           # Si se ha cogido o no la moneda
        done      = False           # Si se ha terminado (llegado al estado final)
        batch_con_recompensa =  False
        observaciones = deque()

        tabla = DQN.calcularTabla()

        GUI.actualizar(tabla)

        for o in range(OBSERVACIONES):

            action, next = DQN.take_action(state)
            GUI.visualize(state, next, o, t,coinObt)

            #if next[:2] == end_state:
            if next == end_state:
                reward = 25 #/ (numMov + 1)
                done = True
                batch_con_recompensa = True
            elif next[:2] == coin_pos and not coinObt:
                coinObt = True
                next[2] = 1
                reward = 15 #/ (numMov + 1)
                coinObtAt = numMov + 1
                batch_con_recompensa = True
            else:
                reward = 0

            observaciones.append((state, action, next, reward, done))

            state = next[:]
            numMov += 1

            if done:
                GUI.visualize(state,init_state,o,t,coinObt)
                print('Finished in %i movements, coin obtained in %i' % (numMov, coinObtAt))
                state     = init_state
                numMov    = 0
                coinObtAt = 0
                coinObt   = False
                done      = False

                DQN.updateExploration(-0.01)

        print('Observation finished')
        GUI.visualize(state,init_state,o,t,coinObt)

        if batch_con_recompensa:
            print ("Aprendiendo...")
            DQN.learn(observaciones)


if __name__ == '__main__':
        DQN = DQN(acciones,alpha,gamma,exploration)
        GUI = Maze(8,8,coin_pos,end_state)
        execute(1000)