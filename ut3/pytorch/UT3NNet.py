import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class SepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        k = min(in_channels, out_channels)
        self.a = nn.Conv2d(k, k, kernel_size, stride, padding, groups=k)
        self.b = nn.Conv2d(in_channels, out_channels, 1, 1)
        if k < in_channels: self.a, self.b = self.b, self.a

    def forward(self, x):
        return self.b(self.a(x))

class DenseConv2d(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        self.conv = SepConv2d(in_channels, growth)
        self.norm = nn.BatchNorm2d(growth)
        self.nlin = nn.ReLU()

    def forward(self, x):
        return torch.cat((x, self.nlin(self.norm(self.conv(x)))), dim=1)

class UT3NNet(nn.Module):
    def __init__(self, game, args, growth=16):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.board_c = game.getBoardChannels()
        self.action_size = game.getActionSize()
        self.args = args

        super(UT3NNet, self).__init__()
        self.drop = nn.Dropout(self.args.dropout)

        self.conv1 = DenseConv2d(self.board_c+0*growth, growth)
        self.conv2 = DenseConv2d(self.board_c+1*growth, growth)
        self.conv3 = DenseConv2d(self.board_c+2*growth, growth)
        self.conv4 = DenseConv2d(self.board_c+3*growth, growth)

        self.out_pi = SepConv2d(self.board_c+4*growth, 1)
        self.out_v = SepConv2d(self.board_c+4*growth, 1,
            kernel_size=(self.board_x, self.board_y), padding=0)

    def forward(self, s):

        print(s.shape)

        s = self.conv1(s)
        s = self.conv2(s)
        s = self.conv3(s)
        s = self.conv4(s)

        s = self.drop(s)

        print(s.shape)

        pi = self.out_pi(s)
        v = self.out_v(s)

        pi = pi.view(-1, self.args.num_channels*self.board_x*self.board_y)

        print(pi.shape, v.shape)

        raise

        return F.log_softmax(pi, dim=1), torch.tanh(v)
