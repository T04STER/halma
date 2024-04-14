
from dataclasses import dataclass
from numbers import Number

import numpy as np
from halma.halma import Halma, Move


@dataclass
class Node:
    move: Move # last move that led to this postion
    board: np.ndarray

class GameTree:
    def __init__(self, halma: Halma) -> None:
        self.halma
        self.start_node = Node(None, self.halma.board)

    def heurestic(self, board: np.ndarray) -> Number:
        """also chcecks win/lose condition (+inf, -inf)"""      

    
    def minmax(self, node: Node, depth: int, maximizing_agent: bool):
        
        if depth == 0 or self.check_terminal_condition(node):
