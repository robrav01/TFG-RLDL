#https://repl.it/NDgQ/4
import random

#Q[][][][0] = 'up', Q[][][][1] = 'down', Q[][][][2] = 'left', Q[][][][3] = 'right'
#Q[][][0] = 'coin not obtained', Q[][][1] = 'coin obtained'
Q = [[[[random.uniform(0.9, 1.1) for i in range(4)] for j in range(2)] for t in range (8)] for s in range (8)]
for x in range(8):
  for y in range(2):
  	Q[0][x][y][0] = -1 #can't go up if you are in the top row
  	Q[7][x][y][1] = -1 #can't go down if you are in the bottom row
  	Q[x][0][y][2] = -1 #can't go left if you are in the leftmost column
  	Q[x][7][y][3] = -1 #can't go right if you are in the rightmost column
  	Q[7][7][y] = [0, 0, 0, 0]

exploration = 1.0
alpha = 0.1
gamma = 0.9 # 0.05

def execute():
  global exploration
  global Q
  state = [0, 0, 0]
  nextState = [0, 0, 0]
  numMov = 0
  coinObtAt = 0

  while (state[:2] != [7, 7]):
    if(random.uniform(0, 1) < exploration):
      action = random.randint(0, 3)
      while (Q[state[0]][state[1]][state[2]][action] < 0):
        action = random.randint(0, 3)
    else:
      action = 0
      actionAux = 1
      value = Q[state[0]][state[1]][state[2]][0]
      for i in Q[state[0]][state[1]][state[2]][1:]:
        if (i > value):
          value = i
          action = actionAux
        actionAux += 1
    
    nextState = state[:]
    if (action == 0):
      nextState[0] -= 1
    elif (action == 1):
      nextState[0] += 1
    elif (action == 2):
      nextState[1] -= 1
    else:
      nextState[1] += 1
    
    if (nextState[:2] == [7, 7]):
      reward = 1/(numMov + 1)
    elif (nextState == [3, 4, 0]):
      reward = 1/(numMov + 1)
      nextState[2] = 1
      coinObtAt = numMov + 1
    else:
      reward = 0

    maxQnext = Q[nextState[0]][nextState[1]][nextState[2]][0]
    for i in Q[nextState[0]][nextState[1]][nextState[2]][1:]:
      if(i > maxQnext):
        maxQnext = i
    
    Q[state[0]][state[1]][state[2]][action] = (1 - alpha)*Q[state[0]][state[1]][state[2]][action] + alpha*(reward + gamma*maxQnext)
    if(Q[state[0]][state[1]][state[2]][action] < 0):
    	Q[state[0]][state[1]][state[2]][action] = 0
    
    state = nextState[:]
    numMov += 1

  if (exploration > 0.05):
    exploration *= 0.9

  print('Finished in %i movements, coin obtained in %i' % (numMov, coinObtAt))

def execute100():
    for i in range(100):
      execute()

def execute1000():
    for i in range(1000):
      print(i)
      execute()

if __name__ == '__main__':
	execute1000()
