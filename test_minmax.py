import math
import time
import numpy as np
from tqdm import tqdm
from halma import get_board
from game_tree import BOARD_SCORE, GameTree
from halma import *
from test_alpha_beta import GameResultTracker, make_random_moves, print_grt


def play_game_minmax(gt: GameTree, depth: int, move_limit: int, grt: GameResultTracker, heuristic_1, heuristic_2):
    start_time = time.perf_counter()
    result = gt.play(depth, True, move_limit, heuristic_1, heuristic_2)
    end_time = time.perf_counter() 
    delta_time =  end_time - start_time
    grt.game_time_list.append(delta_time)
    if result >= float('inf'):
        grt.player_1_wins += 1
        return (1, 0, 0)
    elif result <= float('-inf'):
        grt.player_2_wins += 1
        return (0, 1, 0)
    else:
        grt.ties += 1
        return (0, 0, 1)


def test_game_minmax(halma: Halma, depth: int, move_limit: int, grt: GameResultTracker, heuristic_1, heuristic_2):
    """Test game and test it when sides are reversed"""
    reverse_board = -1 * np.flip(halma.game_state.board, axis=(0,1))
    halma_rev = Halma(get_board())
    halma_rev.game_state = GameState.create_game_state_from_board(reverse_board)

    gt = GameTree(halma)
    game_result = play_game_minmax(gt, depth, move_limit, grt, heuristic_1, heuristic_2)

    gt = GameTree(halma_rev)
    game_result_rev = play_game_minmax(gt, depth, move_limit, grt, heuristic_1, heuristic_2)
    round_result = tuple(map(sum, zip(game_result, game_result_rev)))
    grt.results_per_round.append(round_result)


def test_run_minmax(depth, move_limit, game_count, initial_move_count,  heuristic_1, heuristic_2) -> GameResultTracker:
    grt = GameResultTracker()
    for _ in tqdm(range(game_count)):
        halma = Halma(get_board())
        make_random_moves(halma, initial_move_count)
        test_game_minmax(halma, depth, move_limit, grt, heuristic_1, heuristic_2)
        del halma
    return grt

if __name__ == '__main__':
    move_limit = 500
    depth = 2
    round_count = 5
    grt = test_run_minmax(depth, move_limit, round_count, initial_move_count=40, heuristic_1=3, heuristic_2=2)
    print_grt(grt, move_limit)