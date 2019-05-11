import random

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        while True:
            a = random.randrange(self.game.getActionSize())
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
    def __init__(self, game, depth=2):
        self.game = game
        self.depth = depth
        self.ended = {}
        self.actions = {}

    def search(self, board, depth, alpha, beta):
        state = self.game.stringRepresentation(board)

        if state not in self.ended:
            self.ended[state] = self.game.getGameEnded(board, 1)

        if self.ended[state]:
            return -self.ended[state], None

        if state not in self.actions:
            self.actions[state] = [a for a,x in enumerate(self.game.getValidMoves(board, 1)) if x]

        if depth == 0:
            return -self.ended[state], random.choice(self.actions[state])

        action, value = None, -float('inf')
        for a in self.actions[state]:
            next_board, next_player = self.game.getNextState(board, 1, a)
            next_board = self.game.getCanonicalForm(next_board, next_player)
            v = search(self, next_board, depth-1, -beta, -alpha)
            if v > value: action, value = a, v
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        wins = [(v, a) for v, a in value_action if v == 1]
        if len(wins):
            value, action = random.choice(wins)
            return -value, action

        unknowns = [(v, a) for v, a in value_action if v == 0]
        if len(unknowns):
            value, action = random.choice(unknowns)
            return -value, action

        draws = [(v, a) for v, a in value_action if v > -1]
        if len(draws):
            value, action = random.choice(draws)
            return -value, action

        value, action = random.choice(value_action)
        return -value, action

    def play(self, board):
        return self.search(board, self.depth, -float('inf'), float('inf'))[1]
