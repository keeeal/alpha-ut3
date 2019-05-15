
import torch
from torch import nn

class UT3NNet(nn.Module):
    def __init__(self, game, args, width=16):
        self.size = game.getBoardSize()
        self.channels = game.getBoardChannels()
        self.actions = game.getActionSize()
        self.args = args

        super(UT3NNet, self).__init__()
        #self.drop = nn.Dropout2d(self.args.dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu, self.tanh = nn.ReLU(), nn.Tanh()
        self.norm = nn.BatchNorm1d(4*9*args.width)

        self.conv1 = nn.Conv2d(self.channels, args.width,
            kernel_size=(3,3), stride=3)
        self.conv2 = nn.Conv2d(self.channels, 3*args.width,
            kernel_size=(3,9), stride=3)
        self.conv3 = nn.Conv2d(self.channels, 3*args.width,
            kernel_size=(9,3), stride=3)
        self.conv4 = nn.Conv2d(self.channels, 9*args.width,
            kernel_size=(9,9))

        self.out_pi = nn.Linear(4*9*width, self.actions)
        self.out_v = nn.Linear(4*9*width, 1)

    def forward(self, x):
        x1 = self.conv1(x).view(-1, 9*self.args.width)
        x2 = self.conv2(x).view(-1, 9*self.args.width)
        x3 = self.conv3(x).view(-1, 9*self.args.width)
        x4 = self.conv4(x).view(-1, 9*self.args.width)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.norm(self.relu(x))

        pi = self.out_pi(x)
        v = self.out_v(x)

        return self.log_softmax(pi), self.tanh(v)
