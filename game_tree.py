from dataclasses import dataclass
from numbers import Number
import time
import numpy as np
from halma import *
from scipy.signal import fftconvolve 


BOARD_SCORE_v1 = np.array([[(1 + j - i) for j in range(16)] for i in range(16, 0, -1)])

def init_score_board(halma=Halma(get_board())):
    score = BOARD_SCORE_v1
    player_camp = score.copy() * np.abs(halma.game_state.board)
    score += 2*player_camp
    return score

def init_diag_score():
    diag_score = np.zeros((16,16))

    for i in range(16):
        diag_score[i, i] = 1
        for j in range(1, 5):
            if  0 <= (x:= i+j) < 16:
                diag_score[x, i] = 1 - 0.2*j
            if  0 <= (x:= i-j) < 16:
                diag_score[x, i] = 1 - 0.2*j
    return diag_score

BOARD_SCORE = init_score_board()
DIAG_SCORE = init_diag_score()

# heurestic 3 kernel
KERNEL = np.array([[1.2, 1, 1.2],
                   [1, 0, 1],
                   [1.2, 1, 1.2]])
KS_ = 1 # center of kernel

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
        self.visited_nodes = 0
        self.heuristic = self.heuristic_1


    def check_terminal_condition(self, game_state: GameState):
        winner = self.halma.check_win_condition(game_state)
        if winner == 1:
            return float('inf')
        elif winner == 2:
            return float('-inf')
        return 0

    def heuristic_1(self, game_state: GameState) -> Number:
        """doesn't check win/lose condition (+inf, -inf)"""
        return np.sum(BOARD_SCORE * np.abs(game_state.board)) + (random.random() - 0.5)

    def heuristic_2(self, game_state: GameState) -> Number:
        board = game_state.board
        return np.sum(BOARD_SCORE * np.abs(board) + (0.5* DIAG_SCORE * board)) + (random.random() - 0.5)


    def heuristic_3(self, game_state: GameState) -> Number:
        board = game_state.board
        return np.sum(BOARD_SCORE * np.abs(board)) + 0.25*np.sum(fftconvolve(board, KERNEL, mode='valid')) + (random.random() - 0.5)


    def minmax(self, depth: int, maximizing_agent: bool) -> int:
        self.visited_nodes += 1
        gs = self.halma.game_state
        terminal = self.check_terminal_condition(gs)
        if terminal != 0:
            return terminal
        if depth == 0:
            return self.heuristic(gs)

        next_depth = depth-1
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            max_value = min_inf
            for move in move_list:
                self.halma.make_move(move)
                result = self.minmax(next_depth, False)
                self.halma.revert_move(move)
                max_value = max(max_value, result)
            return max_value
        else:
            move_list = self.halma.get_available_moves(2, gs)
            min_value = max_inf
            for move in move_list:
                self.halma.make_move(move)
                result = self.minmax(next_depth, True)
                self.halma.revert_move(move)
                min_value = min(min_value, result)
            return min_value

    def minmax_dispatcher(self, depth: int, maximizing_agent: bool):
        self.visited_nodes += 1
        gs = self.halma.game_state
        terminal = self.check_terminal_condition(gs)
        if terminal != 0:
            return terminal, None
        if depth == 0:
            return self.heuristic(gs), None

        next_depth = depth-1
        best_move = None
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            max_result = min_inf
            for move in move_list:
                self.halma.make_move(move)
                result = self.minmax(next_depth, False)
                self.halma.revert_move(move)
                if max_result < result:
                    max_result = result
                    best_move = move

            return max_result, best_move
        else:
            move_list = self.halma.get_available_moves(2, gs)
            min_result = max_inf
            for move in move_list:
                self.halma.make_move(move)
                result = self.minmax(next_depth, True)
                self.halma.revert_move(move)
                if min_result > result:
                    min_result = result
                    best_move = move

            return min_result, best_move
       
    def alpha_beta(self, depth: int, maximizing_agent: bool, alpha = min_inf, beta = max_inf):
        self.visited_nodes += 1
        gs = self.halma.game_state
        terminal = self.check_terminal_condition(gs)
        if terminal != 0:
            return terminal
        if depth == 0:
            return self.heuristic(gs)
        next_depth = depth - 1
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            value = min_inf
            for move in move_list:
                self.halma.make_move(move)
                result = self.alpha_beta(
                    next_depth,
                    False,
                    alpha,
                    beta
                )
                value = max(value, result)
                self.halma.revert_move(move)
                alpha = max(alpha, value)
                if value > beta:
                    break
            return value
        else:
            move_list = self.halma.get_available_moves(2, gs)
            value = max_inf
            for move in move_list:
                self.halma.make_move(move)
                result = self.alpha_beta(
                    next_depth,
                    True,
                    alpha,
                    beta
                )
                value = min(value,result)
                self.halma.revert_move(move)
                beta = min(beta, value)
                if value < alpha:
                    break
            return value
        
    def alpha_beta_dispatcher(self, depth: int, maximizing_agent: bool, alpha = min_inf, beta = max_inf):
        self.visited_nodes += 1
        gs = self.halma.game_state
        terminal = self.check_terminal_condition(gs)
        if terminal != 0:
            return terminal, None
        if depth == 0:
            return self.heuristic(gs), None
        
        best_move = None
        next_depth = depth-1
        if maximizing_agent:
            move_list = self.halma.get_available_moves(1, gs)
            value = min_inf
            for move in move_list:
                self.halma.make_move(move)
                result = self.alpha_beta(next_depth, False, alpha, beta)
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
                result = self.alpha_beta(next_depth, True, alpha, beta)
                value, best_move = min((value, best_move), (result, move), key=lambda x:x[0])
                self.halma.revert_move(move)
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value, best_move

    def assign_heuristic(self, player, heuristic_1, heuristic_2):
        if player:
            if heuristic_1 == 1:
                self.heuristic = self.heuristic_1
            elif heuristic_1 == 2:
                self.heuristic = self.heuristic_2
            else:
                self.heuristic = self.heuristic_3 
        else: 
            if heuristic_2 == 1:
                self.heuristic = self.heuristic_1
            elif heuristic_2 == 2:
                self.heuristic = self.heuristic_2
            else:
                self.heuristic = self.heuristic_3 
        

    def play(self, depth=2, player=True, max_count=5, heuristic_1=None, heuristic_2=None):
        win_condition = 0
        counter = 0
        while win_condition != float('-inf') and win_condition != float('inf') and counter < max_count:
            self.assign_heuristic(player, heuristic_1, heuristic_2)

            win_condition, move = self.alpha_beta_dispatcher(
                depth,
                player
            )
            if move is None:
                break
            self.halma.make_move(move)
            player = not player
            counter += 1
        return win_condition
    

    def play_minmax(self, depth=2, player=True, max_count=5, heuristic_1=1, heuristic_2=1):
        win_condition = 0
        counter = 0
        while win_condition != float('-inf') and win_condition != float('inf') and counter < max_count:
            self.assign_heuristic(player, heuristic_1, heuristic_2)
            win_condition, move = self.minmax_dispatcher(
                depth,
                player
            )
            if move is None:
                break
            self.halma.make_move(move)
            player = not player
            counter += 1
        return win_condition


if __name__ == '__main__':
    halma = Halma(get_board())
    gt = GameTree(halma)
    
    st=time.perf_counter()
    bb = gt.play_minmax(depth=1, max_count=1)
    print(f"Elapsed: {time.perf_counter()-st}")
    print(bb)
    halma = Halma(get_board())
    gt = GameTree(halma)
    st=time.perf_counter()
    bb = gt.play(depth=2, max_count=100)
    print(f"Elapsed: {time.perf_counter()-st}")
    print(bb)
    # stats = pstats.Stats('profile_stats')


    # stats.sort_stats('cumulative').print_stats()
