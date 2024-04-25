import pstats
from pstats import SortKey
import cProfile
from dataclasses import dataclass
from functools import cache, lru_cache
from numbers import Number
import random
import time
from typing import Tuple
from concurrent import futures
import numpy as np
from halma import *


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

    def heuristic(self, game_state: GameState) -> Number:
        """doesn't check win/lose condition (+inf, -inf)"""
        self.check_terminal_condition(game_state)
        return np.sum(BOARD_SCORE * np.abs(game_state.board))
  

    def minmax(self, node: Node, depth: int, maximizing_agent: bool, initial_move:Move=None) -> Tuple[int, Move]:
        gs = node.game_state
        heurestic = self.heuristic(gs)
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
            
            max_= max(list(minmax_generator), key=lambda res:res[0])
            return max_
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
            min_ = min(minmax_generator, key=lambda res:res[0])
            return min_
    
    def alpha_beta(self, node: Node, depth: int, maximizing_agent: bool, alpha = min_inf, beta = max_inf):
        gs = self.halma.game_state
        terminal = self.check_terminal_condition(gs)
        if terminal != 0:
            return terminal
        if depth == 0:
            return self.heuristic(gs)

        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            value = min_inf
            for move in move_list:
                self.halma.make_move(move)
                next_node = Node(move, None)
                result = self.alpha_beta(
                    next_node,
                    depth-1,
                    False,
                    alpha,
                    beta
                )
                value = max(value, result)
                self.halma.revert_move(move)
                alpha = max(alpha, value)
                if value >= beta:
                    break
            return value
        else:
            move_list = self.halma.get_available_moves(2, gs)
            value = max_inf
            for move in move_list:
                self.halma.make_move(move)
                next_node = Node(move, None)
                result = self.alpha_beta(
                    next_node,
                    depth-1,
                    True,
                    alpha,
                    beta
                )
                value = min(value,result)
                self.halma.revert_move(move)
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value
        
    def alpha_beta_dispatcher(self, depth: int, maximizing_agent: bool, alpha = min_inf, beta = max_inf):
        gs = self.halma.game_state
        terminal = self.check_terminal_condition(gs)
        if terminal != 0:
            return terminal, None
        if depth == 0:
            return self.heuristic(gs), None
        
        best_move = None
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            value = min_inf
            for move in move_list:
                self.halma.make_move(move)
                next_node = Node(move, None)
                result = self.alpha_beta(
                        next_node,
                        depth-1,
                        False,
                        alpha,
                        beta
                )
                self.halma.revert_move(move)
                value, best_move = max((value, best_move), (result, move), key=lambda x:x[0])
                alpha = max(alpha, value)
                if value >= beta:
                    break
            return value, best_move
        else:
            move_list = self.halma.get_available_moves(2, gs)
            value = max_inf
            for move in move_list:
                self.halma.make_move(move)
                next_node = Node(move, None)
                result = self.alpha_beta(
                        next_node,
                        depth-1,
                        True,
                        alpha,
                        beta
                )
                value, best_move = min((value, best_move), (result, move), key=lambda x:x[0])
                self.halma.revert_move(move)
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value, best_move

    def alpha_beta_pruning_copy(self, node, depth: int, maximizing_agent: bool, initial_move:Move=None, alpha = float('-inf'), beta = float('inf')):
        gs = node.game_state
        heurestic = self.heuristic(gs)
        if depth == 0 or heurestic == float('inf') or heurestic == float('-inf'):
            return heurestic, initial_move
        
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            value = float('-inf')
            for move in move_list:
                next_node = Node(move, self.halma.make_virtual_move(move, gs))
                value, initial_move = max(
                    (value, initial_move),
                    self.alpha_beta(
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
                    self.alpha_beta(
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
            win_condition, move = self.alpha_beta_dispatcher(
                depth,
                player
            )    
            self.halma.make_move(move)
            player = not player
            counter += 1
        print(win_condition)
        return self.halma.game_state.board

if __name__ == '__main__':
    halma = Halma(get_board())
    gt = GameTree(halma)
    
    st=time.time()
    gt.play(depth=4, max_count=2)
    print(f"Elapsed: {time.time()-st}")
    # stats = pstats.Stats('profile_stats')


    # stats.sort_stats('cumulative').print_stats()
