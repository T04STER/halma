
import cProfile
from dataclasses import dataclass
from functools import cache
from numbers import Number
import random
from typing import Tuple

import numpy as np
from halma import *
from utils import timeit


@dataclass
class Node:
    move: Move # last move that led to this postion
    game_state: GameState

class GameTree:
    def __init__(self, halma: Halma) -> None:
        self.halma: Halma = halma
        self.start_node = Node(None, self.halma.game_state)

    def check_terminal_condition(self, game_state: GameState):
        winner = self.halma.check_win_condition(game_state)
        if winner == 1:
            return float('inf')
        elif winner == 2:
            return float('-inf')
        return 0

    @cache
    def heurestic(self, game_state: GameState) -> Number:
        """also checks win/lose condition (+inf, -inf)"""    
        end_game = self.check_terminal_condition(game_state)
        if end_game != 0:
            return end_game
        return random.randint(-100, 100)
    
    def minmax(self, node: Node, depth: int, maximizing_agent: bool, initial_move:Move=None) -> Tuple[int, Move]:
        gs = node.game_state
        heurestic = self.heurestic(gs)
        if depth == 0 or heurestic == float('inf') or heurestic == float('-inf'):
            return heurestic, initial_move
        
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            minmax_generator = (
                self.minmax(
                    Node(move, self.halma.make_virtual_move(move, gs)),
                    depth-1,
                    False,
                    move if not initial_move else initial_move
                )
                for move in move_list
            )
            
            return max(minmax_generator, key=lambda res:res[0])
        else:
            move_list = self.halma.get_available_moves(player=2, game_state=gs)
            minmax_generator = (
                self.minmax(
                    Node(move, self.halma.make_virtual_move(move, gs)),
                    depth-1,
                    True,
                    move if not initial_move else initial_move
                )
                for move in move_list
            )
            return min(minmax_generator, key=lambda res:res[0])
    
    def play(self, depth=2, player=True, max_count=10):
        win_condition = 0
        counter = 0
        while win_condition != float('-inf') and win_condition != float('inf') and counter < max_count:
            win_condition, move = self.minmax(
                Node(None, self.halma.game_state),
                depth,
                player
            )            
            self.halma.make_move(move)
            player = not player
            counter += 1
            
        return self.halma.game_state.board

if __name__ == '__main__':
    halma = Halma(get_board())
    gt = GameTree(halma)
    cProfile.run('gt.play(depth=4, max_count=1)')
