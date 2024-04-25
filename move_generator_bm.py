
import time
from typing import List
import pandas as pd
import numpy as np

from halma import GameState, Halma, get_board


def generate_random_game_state_list(count: int) ->  List[GameState]:
    board = get_board()
    gs_list = []
    for _ in range(count):
        np.random.shuffle(board)
        new_state = GameState.create_game_state_from_board(board)
        gs_list.append(new_state)
    return gs_list


def test_python_move_generation(halma: Halma, game_state_list: List[GameState]):
    result_list = [] 
    for game_state in game_state_list:
        start = time.perf_counter_ns()
        halma.get_available_moves_py(1, game_state)
        end = time.perf_counter_ns()
        result_list.append(end-start)
    return result_list

def test_cpp_powered_move_generation(halma: Halma, game_state_list: List[GameState]):
    result_list = [] 
    for game_state in game_state_list:
        start = time.perf_counter_ns()
        halma.get_available_moves(1, game_state)
        end = time.perf_counter_ns()
        result_list.append(end-start)
    return result_list

if __name__ == "__main__":
    game_state_list = generate_random_game_state_list(1_000_000)
    halma_controller = Halma(get_board()) # get board can be ignored
    time_py = test_python_move_generation(halma_controller, game_state_list)
    time_cpp = test_cpp_powered_move_generation(halma_controller, game_state_list)
 
    del game_state_list
    df_py = pd.DataFrame(time_py)
    
    avg_std_py = df_py.describe().loc[['mean', 'std', '50%']]
    del df_py
    df_cpp = pd.DataFrame(time_cpp)
    avg_std_cpp = df_cpp.describe().loc[['mean', 'std', '50%']]
    del df_cpp
    result_df = pd.concat([avg_std_py, avg_std_cpp], axis=1)
    result_df.columns = ['Python (avg, std, median)', 'C++ (avg, std, median)']
    print(result_df)
    print("\n")

    latex_table = result_df.to_latex()
    print(latex_table)