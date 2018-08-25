import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from NeuralNetwork import DQN

from collections import deque

import gym

################################
#   GLOBAL CONSTANTS
################################

# Numero de movimientos observados antes de aprender
OBSERVACIONES = 300

exploration = 1.0
alpha = 0.05
gamma = 0.9

acciones = [0, 1]

def execute(times, env):
    for t in range(times):
        print(t)

        # INICIALIZACION
        state     = env.reset()
        done      = False           # Si se ha terminado (llegado al estado final)
        observaciones = deque()

        for o in range(OBSERVACIONES):

            action = DQN.take_action(state)

            if t > 20:
                env.render()
            next_state, reward, done, hola = env.step(action)

            if done:
                reward = -10

            observaciones.append((state, action, next_state, reward, done))

            state = next_state[:]

            if done:
                state     = env.reset()
                done      = False

                DQN.updateExploration(-0.01)


        DQN.learn(observaciones)


if __name__ == '__main__':
        DQN = DQN(acciones, alpha, gamma, exploration)
        env = gym.make('CartPole-v1')
        execute(100, env)
