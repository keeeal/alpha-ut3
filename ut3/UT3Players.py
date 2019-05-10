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
        valid = self.game.getValidMoves(board, 1)
        print('Valid moves:')
        print(', '.join(str(int(i/self.game.n**2))+' '+str(int(i%self.game.n**2))
            for i, v in enumerate(valid) if v))
        while True:
            a = input()
            x, y = [int(x) for x in a.split(' ')]
            a = self.game.n**2 * x + y
            if valid[a]:
                break
            else:
                print('Invalid')
        return a


class MinMaxUT3Player():
    def __init__(self, game):
        self.game = game
        self.end = {}

    def search(self, board, depth):
        key = self.game.stringRepresentation(board)

        if key not in self.end:
            self.end[key] = self.game.getGameEnded(board, 1)

        if self.end[key] or depth == 0:
            return -self.end[key], None

        value_action = []

        for a, valid in enumerate(self.game.getValidMoves(board, 1)):
            if valid:
                next_board, next_player = self.game.getNextState(board, 1, a)
                next_board = self.game.getCanonicalForm(next_board, next_player)
                value_action.append((self.search(next_board, depth-1)[0], a))

        value, action = max(value_action)
        return -value, action

    def play(self, board):
        return self.search(board, 2)[1]
