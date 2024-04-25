import time
import numpy as np
from halma import get_board
from game_tree import BOARD_SCORE, GameTree
from halma import *



if __name__ == '__main__':
    halma = Halma(get_board())
    gt = GameTree(halma)
        
    st=time.time()
    gt.play(depth=3, max_count=300)
    print(f"Elapsed: {time.time()-st}")
    print(gt.halma.game_state.board)