from dataclasses import dataclass
from utils import timeit
from typing import List, Literal, Optional, Set, Tuple, Union

import numpy as np

@dataclass
class Move:
    src: Tuple[int, int]
    dest: Tuple[int, int]

class Halma:
    """
        Class representing game   
    """
    BOARD_SIZE = 16
    def __init__(self, board: np.ndarray) -> None:
        self.board = board
        self.player1_camp = set(tuple(p) for p in np.dstack(np.where(board==1))[0])
        self.player2_camp = set(tuple(p) for p in np.dstack(np.where(board==-1))[0])
        
        
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
        
    def get_available_moves(self, player: Literal['1', '2'], board: np.ndarray=None) -> List[Move]:
        moves = []
        if board is None:
            board = self.board
        pawn_list = self.get_pawn_list(player, board)

        for pos in pawn_list:
            moves.extend(self.get_pawn_moves(pos, board))
        return  moves

    def make_move(self, move: Move):
        xs, ys = move.src 
        xd, yd = move.dest
        self.board[xd][yd] = self.board[xs][ys]
        self.board[xs][ys] = 0

    def make_virtual_move(self, move, board):
        "returns a copy of board with made move"
        xs, ys = move.src 
        xd, yd = move.dest
        board = board.copy()
        board[xd][yd] = board[xs][ys]
        board[xs][ys] = 0
        return board
    
    def check_win_condition(self, board=None) -> Literal['0', '1', '2']:
        board = board if board is None else self.board
        
        player_1 = set(tuple(p) for p in self.get_pawn_list(1, board))
        if player_1 == self.player2_camp:
            return 1
        
        player_2 = set(tuple(p) for p in self.get_pawn_list(2, board))
        if player_2 == self.player1_camp:
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
    print(pawns:=halma.get_pawn_list(1, halma.board))
