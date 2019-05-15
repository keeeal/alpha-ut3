
import sys
sys.path.append('..')
from Game import Game
from .UT3Logic import Board
import numpy as np

class UT3Game(Game):
    def __init__(self, n=3):
        self.n = n

    def getArray(self, b):
        macro = np.tile(b.macro, (self.n, self.n))
        return np.stack((b.pieces, macro))

    def getBoardChannels(self):
        return 2

    def getBoardSize(self):
        return self.n**2, self.n**2

    def getActionSize(self):
        return self.n**4

    def getInitBoard(self):
        b = Board(self.n)
        return self.getArray(b)

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1,:3,:3])
        move = int(action/self.n**2), action%self.n**2
        b.execute_move(move, player)
        return self.getArray(b), -player

    def getValidMoves(self, board, player):
        valid = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1,:3,:3])
        for x, y in b.get_legal_moves(player):
            valid[x*self.n**2 + y] = 1
        return np.array(valid)

    def getGameEnded(self, board, player):
        #Return 0 if not ended, 1 if player 1 won, -1 if player 1 lost.
        #Return small non-zero value for a draw.
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1,:3,:3])
        for player in -1, 1:
            if b.is_win(player):
                return player
            if b.is_full():
                return b.draw
        return 0

    def getCanonicalForm(self, board, player):
        return np.where(board, player*board, board)

    def getSymmetries(self, board, pi):
        # rotate, mirror
        assert(len(pi) == self.getActionSize())  # 1 for pass
        pi_board = np.reshape(pi, self.getBoardSize())
        sym, x, y = [], -2, -1

        for rot in range(4):
            for flip in True, False:
                newB = np.rot90(board, rot, (x, y))
                newPi = np.rot90(pi_board, rot, (x, y))
                if flip:
                    newB = np.flip(newB, y)
                    newPi = np.flip(newPi, y)
                sym.append((newB, list(newPi.ravel())))
        return sym

    def stringRepresentation(self, board):
        return board.tostring()

def display(board, indent='  '):
    print('')
    print(indent + '   0 | 1 | 2 ‖ 3 | 4 | 5 ‖ 6 | 7 | 8')
    print('')
    for n, row in enumerate(board[0]):
        if n:
            if n % 3:
                sep = '---+---+---'
                print(indent + '- ' + sep + '‖' + sep + '‖' + sep)
            else:
                sep = '==========='
                print(indent + '= ' + sep + '#' + sep + '#' + sep)
        row = ' ‖ '.join(' | '.join(map(str, map(int, row[i:i+3]))) for i in range(0, len(row), 3))
        print(indent + str(n) + '  ' + row.replace('-1','O').replace('1','X').replace('0','.'))
    print('')
