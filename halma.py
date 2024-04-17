from copy import copy
from dataclasses import dataclass
from utils import timeit
from typing import List, Literal, Optional, Set, Tuple, Union

import numpy as np

@dataclass
class Move:
    src: Tuple[int, int]
    dest: Tuple[int, int]

class GameState:
    def __init__(self, board: np.ndarray, player_1: Set[Tuple[int, int]], player_2: Set[Tuple[int, int]]) -> None:
        self.board = board
        self.player_1: Set[Tuple[int, int]] = player_1
        self.player_2: Set[Tuple[int, int]] = player_2
    
    def make_move(self, move: Move):
        xs, ys = move.src 
        xd, yd = move.dest
        player = self.board[xs][ys]
        self.board[xd, yd] = player
        if player == 1:
            self.player_1.remove((xs, ys))
            self.player_1.add((xd, yd))
        else:
            self.player_2.remove((xs, ys))
            self.player_2.add((xd, yd))
    
    def copy(self) -> 'GameState':
        return GameState(
            self.board.copy(),
            copy(self.player_1),
            copy(self.player_2)
        )

    def new_state(self, move: Move):
        gs =  self.copy()
        gs.make_move(move)
        return gs

    @classmethod
    def create_game_state_from_board(cls, board:np.ndarray):
        player_1 = set(tuple(p) for p in np.dstack(np.where(board==1))[0])
        player_2 = set(tuple(p) for p in np.dstack(np.where(board==-1))[0])
        return cls(board, player_1, player_2)
    

class Halma:
    """
        Class representing game   
    """
    BOARD_SIZE = 16
    def __init__(self, board: np.ndarray) -> None:
        self.game_state: GameState = GameState.create_game_state_from_board(board)
        self.player1_camp: Set[Tuple[int, int]] = copy(self.game_state.player_1)
        self.player2_camp: Set[Tuple[int, int]]= copy(self.game_state.player_2)
        
    def _is_in_board(self, x, y):
        return (0 <= x < self.BOARD_SIZE) and (0 <= y < self.BOARD_SIZE)

    def get_jump_moves(self, pos, board:np.ndarray, visited:Set[Tuple[int, int]]=set()):
        x, y = pos
        if not self._is_in_board(x, y) or board[x][y] != 0:
            return []
        visited.add(pos)
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if self._is_in_board(i, j) and board[i][j] != 0 and (x, y) != (i, j):
                    dest = self.get_jump_dest(pos, (i,j))
                    if dest not in visited:
                        self.get_jump_moves(dest, board, visited)

        return list(visited)
    
    def get_jump_dest(self, src, over) -> Tuple[int, int]:
        return 2*over[0] - src[0], 2*over[1] - src[1]

    def get_pawn_moves(self, position, board) -> List[Move]:
        # TODO: add not outside camp lock
        x, y = position
        moves_tup = []
        for i in range(max(0, x - 1), min(self.BOARD_SIZE, x+2)):
            for j in range(max(0, y-1), min(self.BOARD_SIZE, y+2)):
                if (i, j) != (x,y):
                    if board[i][j] == 0:
                        moves_tup.append((i,j))
                    else:
                        dest = self.get_jump_dest(position, (i,j))
                        jump_moves = self.get_jump_moves(dest, board, set())
                        moves_tup.extend(jump_moves)
        return [Move(position, dest)  for dest in moves_tup]

    def get_pawn_list(self, player: Literal['1', '2'], board: np.ndarray) -> np.ndarray:
        player_pawn = player if player == 1 else -1
        return np.dstack(np.where(board==player_pawn))[0]
        
    def get_available_moves(self, player: Literal['1', '2'], game_state:Optional[GameState]=None) -> List[Move]:
        moves = []
        if game_state is None:
            game_state = self.game_state
        pawn_list = game_state.player_1 if player == 1 else game_state.player_2

        for pos in pawn_list:
            moves.extend(self.get_pawn_moves(pos, game_state.board))
            
        return  moves

    def make_move(self, move: Move):
        self.game_state.make_move(move)

    def make_virtual_move(self, move: Move, game_state: GameState) -> GameState:
        "returns a copy of gamestate after move"
        return game_state.new_state(move)

    
    def check_win_condition(self, game_state:Optional[GameState]=None) -> Literal['0', '1', '2']:
        if not game_state:
            game_state = self.game_state

        if game_state.player_1 == self.player2_camp:
            return 1
        
        if game_state.player_2 == self.player1_camp:
            return 2
        
        return 0

    def print_board(self):
        print('='*16)
        for i in range(len(self.board)):
            for v in self.board[i]:
                v = 2 if v ==-1  else v
                print(v, end=' ')
            print('')
        
def get_board():
    board = np.zeros((16,16), dtype=np.int8)
    def fill_player1():
        for i in range(5):
            board[0][i] = 1
            board[1][i] = 1

        for i in range(4):
            board[2][i] = 1
        for i in range(3):
            board[3][i] = 1
        for i in range(2):
            board[4][i] = 1
    
    def fill_player2():
        for i in range(5):
            board[15][15-i] = -1
            board[14][15-i] = -1

        for i in range(4):
            board[13][15-i] = -1
        for i in range(3):
            board[12][15-i] = -1
        for i in range(2):
            board[11][15-i] = -1
    
    fill_player1()
    fill_player2()

    return board

if __name__ == '__main__':
    halma = Halma(get_board())
    print(pawns:=halma.get_pawn_list(1, halma.game_state.board))
