import numpy as np

from keras.utils import to_categorical

def one_hot(state):
    x = state[0]
    y = state[1]
    coin = state[2]
    return np.append(np.append([to_categorical(x,8)],[to_categorical(y,8)]),coin)

# No valido para arrays de estados
def decimal(x):
    return [np.argmax(p) for p in x]
