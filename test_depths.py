

from numbers import Number
import time
from typing import Tuple

import numpy as np
from tqdm import tqdm

from game_tree import GameTree
from halma import Halma, get_board
from test_alpha_beta import make_random_moves




def test_depth(gt, depth, heurestic) -> Tuple[Number, Number]:
    st = time.perf_counter()
    bb = gt.play(depth, max_count=1)
    delta = time.perf_counter()-st
    return delta, gt.visited_nodes


def init_game_tree():
    halma = Halma(get_board())
    make_random_moves(halma, 10)
    return GameTree(halma)
    

def run():
    for heurestic in range(1,4):
        heur = []
        heur_nodes = []
        for depth in range(1,5):
            times = []
            node_count = []
            for i in tqdm(range(5)):
                gt = init_game_tree()
                dt, nc = test_depth(gt, depth, heurestic)
                times.append(dt)
                node_count.append(nc)
            print('-'*20)
            print(f"Results depth {depth}, heurestic {heurestic}")
            print(times)
            print("avg time, std")
            tm = np.mean(times)
            tstd = np.std(times)
            print(f"{tm}, {tstd}")
            print("avg node count, std")
            nc =np.mean(node_count)
            ncstd=np.std(node_count)
            print(f"{nc}, {ncstd}")
            heur.append((tm, tstd))
            heur_nodes.append((nc, ncstd))
        print(f"\n\nRESULTS FOR heuristic {heurestic}")
        print(heur)
        print(heur_nodes)
            
            
if __name__ == "__main__":
    run()
                
