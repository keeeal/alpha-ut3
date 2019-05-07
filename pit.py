import Arena
from MCTS import MCTS
from ut3.UT3Game import UT3Game, display
from ut3.UT3Players import *
from ut3.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = UT3Game(6)

# all players
rp = RandomPlayer(g).play
gp = GreedyUT3Player(g).play
hp = HumanUT3Player(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/UT3/pytorch/','6x100x25_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(2, verbose=True))
