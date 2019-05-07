import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        while True:
            a = np.random.randint(self.game.getActionSize())
            if valid[a]: return a


class HumanUT3Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        pass


class GreedyUT3Player():
    '''Always try to win the current microboard.'''
    def __init__(self, game):
        self.game = game

    def play(self, board):
        pass
