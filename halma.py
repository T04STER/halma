from dataclasses import dataclass
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
        self.player1 = []
        self.player2 = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                pawn = board[i][j]
                if pawn == 1:
                    self.player1.append((i,j))
                elif pawn == -1:
                    self.player2.append((i,j))
    
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
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if self._is_in_board(i, j):
                    if board[i][j] == 0:
                        moves_tup.append((i,j))
                    else:
                        dest = self.get_jump_dest(position, (i,j))
                        jump_moves = self.get_jump_moves(dest, board, set())
                        moves_tup.extend(jump_moves)
        return [Move(position, dest)  for dest in moves_tup]

    def get_availible_moves(self, player: Literal['1', '2'], board=None) -> List[Move]:
        pawn_list = self.player1 if player == 1 else self.player2
        moves = []
        if not board:
            board = self.board

        for pos in pawn_list:
            moves.extend(self.get_pawn_moves(pos, board))
        return  moves

    def make_move(self, move: Move):
        xs, ys = move.src 
        xd, yd = move.dest
        self.board[xd][yd] = self.board[xs][ys]
        self.board[xs][ys] = 0

    def make_virtual_move(self, move):
        "returns a copy of board with made move"
        xs, ys = move.src 
        xd, yd = move.dest
        board = self.board.copy()
        board[xd][yd] = board[xs][ys]
        board[xs][ys] = 0
        return board

def get_board():
    board = np.zeros((16,16), dtype=int)
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
    