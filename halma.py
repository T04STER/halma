from copy import copy
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Tuple, Union
import jump_moves as c_util_jump
from functools import lru_cache

import numpy as np

@dataclass
class Move:
    src: Tuple[int, int]
    dest: Tuple[int, int]
    def __hash__(self):
        return hash(self.src)+hash(self.dest)
    def __str__(self) -> str:
        return f"{self.src} -> {self.dest}"

class GameState:
    def __init__(self, board: np.ndarray, player_1: Set[Tuple[int, int]], player_2: Set[Tuple[int, int]]) -> None:
        self.board = board
        self.player_1: Set[Tuple[int, int]] = player_1
        self.player_2: Set[Tuple[int, int]] = player_2

    def __str__(self) -> int:
        return str(self.player_1) + str(self.player_2)

    def make_move(self, move: Move):
        player = self.board[move.src]
        self.board[move.src] = 0
        self.board[move.dest] = player
        if player == 1:
            self.player_1.remove(move.src)
            self.player_1.add(move.dest)
        elif player == -1:
            self.player_2.remove(move.src)
            self.player_2.add(move.dest)
        else:
            raise ValueError("WTF??")

    def copy(self) -> 'GameState':
        return GameState(
            self.board.copy(),
            copy(self.player_1),
            copy(self.player_2)
        )

    def new_state(self, move: Move):
        gs = self.copy()
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
        return (0 <= x < 16) and (0 <= y < 16)

    def get_jump_moves(self, pos, board:np.ndarray):
        return c_util_jump.jump_moves(board, pos)
    
    def get_jump_moves_py(self, pos, board:np.ndarray, visited:Set[Tuple[int, int]]=set()):
        """Deprecated due to c module"""
        x, y = pos
        if not self._is_in_board(x, y) or board[x][y] != 0:
            return []
        queue = [pos]
        while queue:
            current = queue.pop()
            if current not in visited:
                visited.add(current)
                possible_jumps = list(self._get_possible_jumps(x, y, board))
                queue.extend(possible_jumps)
        return list(visited)

    def get_jump_dest(self, src, over) -> Tuple[int, int]:
        return 2*over[0] - src[0], 2*over[1] - src[1]

    def _get_possible_jumps(self, x, y, board):
        xt, xt2 = x+1, x+2
        xb, xb2 = x-1, x-2
        yl, yl2 = y-1, y-2
        yr, yr2 = y+1, y+2
        possible_neighbours = {
            (xt, yl): (xt2, yl2), (xt, y): (xt2, y), (xt, yr): (xt2, yr2),
            (x, yl): (x, yl2), (x, yr): (x, yr2),
            (xb, yl): (xb2, yl2), (xb, y): (xb2, y), (xb, yr): (xb2, yr2)
        }

        jump_dest_gen = (
            jump_dest
            for neighbour, jump_dest in possible_neighbours.items()
            if self._is_in_board(*neighbour)
                and board[neighbour] != 0
                and self._is_in_board(*jump_dest)
                and board[jump_dest] == 0
        )
        return jump_dest_gen


    def _get_neighbours(self, x, y) -> Tuple[Tuple[int, int]]:
        xt, xt2 = x+1, x+2
        xb, xb2 = x-1, x-2
        yl, yl2 = y-1, y-2
        yr, yr2 = y+1, y+2
        possible_neighbours = {
            (xt, yl): (xt2, yl2), (xt, y): (xt2, y), (xt, yr): (xt2, yr2),
            (x, yl): (x, yl2), (x, yr): (x, yr2),
            (xb, yl): (xb2, yl2), (xb, y): (xb2, y), (xb, yr): (xb2, yr2)
        }

        neighbours = {
            neighbour:jump_dest
            for neighbour, jump_dest in possible_neighbours.items()
            if self._is_in_board(*neighbour)
        }
        return neighbours

    def get_pawn_moves(self, position, board) -> List[Move]:
        # TODO: add not outside camp lock
        pos_x, pos_y = position
        move_set = set()
        possition_list = self._get_neighbours(pos_x, pos_y)
        
        for n, dest in possition_list.items():
            if board[n] == 0:
                move_set.add(n)
            else:
                jump_moves = self.get_jump_moves(dest, board)
                move_set.update(jump_moves)
        return [Move(position, dest)  for dest in move_set]

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

    @lru_cache(1024)
    def make_virtual_move(self, move: Move, game_state: GameState) -> GameState:
        "returns a copy of gamestate after move"
        return game_state.new_state(move)

    def revert_move(self, move: Move):
        self.game_state.make_move(Move(src=move.dest, dest=move.src))


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
        for i in range(len(self.game_state.board)):
            for v in self.game_state.board[i]:
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
    board = np.zeros(shape=(16,16), dtype=np.int8)
    board[2,2] = 1
    board[2,3] = 1
    board[3,4] = 1


    halma = Halma(get_board())
    #print(len()
    moves  = halma.get_pawn_moves((2,3), board=halma.game_state.board)
    for move in moves:
        print(move)
    print('='*16)
    dests = [move.dest for move in moves]
    for i, r in enumerate(halma.game_state.board):
        print(i, end=' ')
    print('')
    for i, r in enumerate(halma.game_state.board):
        for j, v in enumerate(halma.game_state.board[i]):
            v = 2 if v ==-1  else v
            if (i,j) in dests:
                v = 'x'
            if (i,j) == (2, 3):
                v = '+'
            print(v, end=' ')
        print('')

    halma.make_move(moves[0])
    moves  = halma.get_available_moves(1)
    dests = [move.dest for move in moves]
    for i, r in enumerate(halma.game_state.board):
        print(i, end=' ')
    print('')
    for i, r in enumerate(halma.game_state.board):
        for j, v in enumerate(halma.game_state.board[i]):
            v = 2 if v ==-1  else v
            if (i,j) in dests:
                v = 'x'
            if (i,j) == (2, 3):
                v = '+'
            print(v, end=' ')
        print('')