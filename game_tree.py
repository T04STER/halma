import pstats
from pstats import SortKey
import cProfile
from dataclasses import dataclass
from functools import cache, lru_cache
from numbers import Number
import random
from typing import Tuple

import numpy as np
from halma import *
from utils import timeit


BOARD_SCORE = np.array([[(1 + j - i) for j in range(16)] for i in range(16, 0, -1)])

max_inf = float('inf')
min_inf = float('-inf')

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

    def heurestic(self, game_state: GameState) -> Number:
        """doesn't check win/lose condition (+inf, -inf)"""      
        return np.sum(BOARD_SCORE * game_state.board)
    
    def minmax(self, node: Node, depth: int, maximizing_agent: bool, initial_move:Move=None) -> Tuple[int, Move]:
        gs = node.game_state
        heurestic = self.heurestic(gs)
        if depth == 0 or heurestic == max_inf or heurestic == min_inf:
            return heurestic, initial_move
        next_depth = depth-1
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            minmax_generator = (
                self.minmax(
                    Node(move, self.halma.make_virtual_move(move, gs)),
                    next_depth,
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
                    next_depth,
                    True,
                    move if not initial_move else initial_move
                )
                for move in move_list
            )
            return min(minmax_generator, key=lambda res:res[0])
    
    def alpha_beta_pruning(self, node, depth: int, maximizing_agent: bool, initial_move:Move=None, alpha = min_inf, beta = max_inf):
        gs = self.halma.game_state
        terminal = self.check_terminal_condition(gs)
        if terminal != 0:
            return terminal, initial_move
        if depth == 0:
            return self.heurestic(gs), initial_move

        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            value = min_inf
            for move in move_list:
                self.halma.make_move(move)
                next_node = Node(move, None)
                value, initial_move = max(
                    (value, initial_move),
                    self.alpha_beta_pruning(
                        next_node,
                        depth-1,
                        True,
                        move if not initial_move else initial_move,
                        alpha,
                        beta
                    ),
                    key=lambda res:res[0],
                )
                self.halma.revert_move(move)

                if value > beta:
                    break
                alpha = max(alpha, value)

            return value, initial_move
        else:
            move_list = self.halma.get_available_moves(2, gs)
            value = max_inf
            for move in move_list:
                self.halma.make_move(move)
                next_node = Node(move, None)
                value, initial_move = min(
                    (value, initial_move),
                    self.alpha_beta_pruning(
                        next_node,
                        depth-1,
                        False,
                        move if not initial_move else initial_move,
                        alpha,
                        beta
                    ),
                    key=lambda res:res[0],
                )
                self.halma.revert_move(move)

                if value < alpha:
                    break
                beta = min(beta, value)
            return value, initial_move
        
    def alpha_beta_pruning_copy(self, node, depth: int, maximizing_agent: bool, initial_move:Move=None, alpha = float('-inf'), beta = float('inf')):
        gs = node.game_state
        heurestic = self.heurestic(gs)
        if depth == 0 or heurestic == float('inf') or heurestic == float('-inf'):
            return heurestic, initial_move
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            value = float('-inf')
            for move in move_list:
                next_node = Node(move, self.halma.make_virtual_move(move, gs))
                value, initial_move = max(
                    (value, initial_move),
                    self.alpha_beta_pruning(
                        next_node,
                        depth-1,
                        False,
                        move if not initial_move else initial_move,
                        alpha,
                        beta
                    ),
                    key=lambda res:res[0],
                )
                if value > beta:
                    break
                alpha = max(alpha, value)

            return value, initial_move
        else:
            move_list = self.halma.get_available_moves(2, gs)
            value = float('inf')
            for move in move_list:
                next_node = Node(move, self.halma.make_virtual_move(move, gs))
                value, initial_move = min(
                    (value, initial_move),
                    self.alpha_beta_pruning(
                        next_node,
                        depth-1,
                        False,
                        move if not initial_move else initial_move,
                        alpha,
                        beta
                    ),
                    key=lambda res:res[0],
                )
                if value < alpha:
                    break
                beta = min(beta, value)
            return value, initial_move

    def play(self, depth=2, player=True, max_count=10):
        win_condition = 0
        counter = 0
        while win_condition != float('-inf') and win_condition != float('inf') and counter < max_count:
            win_condition, move = self.alpha_beta_pruning(
                None,
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
    # gt.play(depth=3, max_count=3)
    cProfile.run('gt.play(depth=3, max_count=5)')
    # stats = pstats.Stats('profile_stats')


    # stats.sort_stats('cumulative').print_stats()
