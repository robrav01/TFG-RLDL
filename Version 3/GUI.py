import tkinter
import random
import time
import numpy as np

GRID_SIZE = 50     # Pixels per grid
ORIGIN = 100       # Determines the position of the maze (just for GUI)

# Visualizar a partir de
timesBeforeGUI = 20

class Maze(tkinter.Tk, object):

   def __init__(self, width, height, coin):
       super(Maze, self).__init__()
       self.width = width
       self.height = height

       # Coordinates of each grid as tuple of the form (x,y)
       self.grid = []

       self.title('Maze {dim1} x {dim2}'.format(dim1=self.width, dim2=self.height))
       self.geometry('{dim1}x{dim2}'.format(dim1=2*ORIGIN + self.width * GRID_SIZE,
                                            dim2=2*ORIGIN + self.height * GRID_SIZE))

       self.coin = (ORIGIN + coin[0]*GRID_SIZE,ORIGIN + coin[1]*GRID_SIZE)
       self._create_canvas()

       self.textos = [[[self.canvas.create_text(ORIGIN + GRID_SIZE * i + (GRID_SIZE/5 * (k+1)),ORIGIN + GRID_SIZE * j + (GRID_SIZE/5 * (k+1))) for k in range(2)] for j in range(8)] for i in range(8)]
       self.valores = {0: 'l', 1: 'r', 2: 'u', 3: 'd'}

   def visualize(self,state,next,o,t,coin_obt):
       if t > timesBeforeGUI:
           self._move_player((state[0] * GRID_SIZE,state[1]*GRID_SIZE),(next[0]*GRID_SIZE,next[1]*GRID_SIZE))
           self.canvas.itemconfigure(self.textoObservaciones,text='Observaciones: %i' % o)
           self.canvas.itemconfigure(self.textoEpochs, text='Epochs: %i' % t)

           if coin_obt:
               self.canvas.itemconfigure(self.coin_item,state=tkinter.HIDDEN)
           else:
               self.canvas.itemconfigure(self.coin_item, state=tkinter.NORMAL)

   def actualizar(self,tabla):
       for i in range(8):
           for j in range(8):
               for c in range(2):
                   texto = self.valores.get(tabla[i][j][c])
                   self.canvas.itemconfigure(self.textos[i][j][c],text=texto)


   def _create_canvas(self):
       self.canvas = tkinter.Canvas(self, bg = 'white',
                                    width =self.width * GRID_SIZE,
                                    height =self.height * GRID_SIZE,
                                    )
       # Draw the grids (and store their coords. Each grid is identified by its upper left corner)
       for row in range(ORIGIN, self.height*GRID_SIZE+ORIGIN, GRID_SIZE):
           for col in range(ORIGIN, self.width*GRID_SIZE +ORIGIN, GRID_SIZE):
               self.grid.append((col,row))
               self.canvas.create_line(col, row, col + GRID_SIZE, row, fill='gray')
               self.canvas.create_line(col, row, col, row + GRID_SIZE, fill='gray')

       # Draw the frame
       self.canvas.create_line(ORIGIN, ORIGIN, ORIGIN, GRID_SIZE * self.width + ORIGIN);
       self.canvas.create_line(ORIGIN, GRID_SIZE * self.width + ORIGIN,
                               GRID_SIZE * self.height + ORIGIN, GRID_SIZE * self.width + ORIGIN)
       self.canvas.create_line(GRID_SIZE * self.height + ORIGIN, GRID_SIZE * self.width + ORIGIN,
                               GRID_SIZE * self.height + ORIGIN, ORIGIN)
       self.canvas.create_line(ORIGIN, ORIGIN, GRID_SIZE * self.height + ORIGIN, ORIGIN);

       self.textoObservaciones = self.canvas.create_text(2*ORIGIN,self.height*GRID_SIZE + ORIGIN + 20,text='Observaciones: %i' % 0)
       self.textoEpochs        = self.canvas.create_text(2*ORIGIN,self.height*GRID_SIZE + ORIGIN + 40,text='Epochs: %i' % 0)

       self._create_escape()

       self._create_player()

       self.create_coin()

       self.canvas.pack(fill='both',expand='yes')

   def _create_player(self):
       self.initial_grid = 0
       self.player = self.grid[self.initial_grid]

       self.player_item = self.canvas.create_oval(
           np.array(self.player)[0] + GRID_SIZE / 4,
           np.array(self.player)[1] + GRID_SIZE / 4,
           np.array(self.player)[0] + 3 * (GRID_SIZE / 4),
           np.array(self.player)[1] + 3 * (GRID_SIZE / 4), fill='red')

   def _create_escape(self):
       self.escape = self.grid[(self.width * self.height)-1]
       point = np.array(self.escape)
       self.canvas.create_rectangle(point[0], point[1],
                                    point[0] + GRID_SIZE, point[1] + GRID_SIZE, fill='orange')

   def create_coin(self):
       self.coin_item = self.canvas.create_oval(
           np.array(self.coin)[0] + 1.5 * (GRID_SIZE / 4),
           np.array(self.coin)[1] + 1.5 * (GRID_SIZE / 4),
           np.array(self.coin)[0] + 2.5 * (GRID_SIZE / 4),
           np.array(self.coin)[1] + 2.5 * (GRID_SIZE / 4), fill='yellow')

   def _move_player(self,state,next_state):
       despl_x = np.array(next_state)[0] - np.array(state)[0]
       despl_y = np.array(next_state)[1] - np.array(state)[1]
       self.canvas.move(self.player_item, despl_x, despl_y)
       self.player = next_state
       self.refresh()

   def refresh(self):
       time.sleep(0.1)
       self.update()

   def reset(self):
       self.refresh()
       self._move_player(self.player,self.grid[self.initial_grid])

       return self.player