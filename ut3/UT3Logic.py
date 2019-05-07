'''
Author: James Keal
Date: May 5, 2019
Board class.
Macro board data:
  1.=X, -1.=O, 0.=empty, -0.=blocked
Pieces board data:
  1.=X, -1.=O, 0.=empty
'''

from itertools import product
import numpy as np

class Board:
    def __init__(self, n=3):
        # Create empty macro and board arrays
        self.n = n
        self.macro = np.zeros((self.n, self.n))
        self.pieces = np.zeros((self.n**2, self.n**2))

    def __getitem__(self, index):
        return self.pieces[index]

    def get_microboard(self, index):
        return self[tuple(slice(self.n*i, self.n*(i+1)) for i in index)]

    def get_legal_moves(self, player):
        """Returns all legal moves for a given player."""
        moves = []
        for u in product(range(self.n), range(self.n)):
            if not self.macro[u] and 0 < np.copysign(1, self.macro[u]):
                for move in product(*(range(self.n*i, self.n*(i+1)) for i in u)):
                    if self.pieces[move] == 0:
                        moves.append(move)
        return moves

    def is_win(self, player, board=None):
        """Check whether a given player has a line in any direction."""
        if board is None: board = self.macro
        for i in range(len(board)):
            if all(board[i,:] == player): return True
            if all(board[:,i] == player): return True
        if all(board.diagonal() == player): return True
        if all(board[::-1].diagonal() == player): return True
        return False

    def execute_move(self, move, player):
        """Place a piece on the board and update the macro."""
        _u = tuple(int(i/self.n) for i in move)
        _v = tuple(int(i%self.n) for i in move)
        assert self.pieces[move] == 0
        self.pieces[move] = player

        for player in -1, 1:
            if self.is_win(player, self.get_microboard(_u)):
                self.macro[_u] = player

        for u in product(range(self.n), range(self.n)):
            if not self.macro[u]:
                self.macro[u] = 0. if self.macro[_v] or u == _v else -0.
