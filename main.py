import numpy as np
from halma import get_board
import jump_moves

board = np.zeros(shape=(16,16), dtype=np.int8)
board[2,2] = 1
result = jump_moves.jump_moves(board, (2, 0))
print(result)
