import numpy as np
from halma import get_board
import jump_moves

rows = 16
cols = 16

matrix = [[1 + j - i  for j in range(cols)] for i in range(rows, 0, -1)]

n = np.array(matrix)
print(n)
print(n.shape)