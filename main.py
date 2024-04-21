import numpy as np
from halma import get_board
import jump_moves

BOARD_SCORE = np.array([[(1 + j - i) * 4  for j in range(16)] for i in range(16, 0, -1)])

for i in BOARD_SCORE:
    print(i)