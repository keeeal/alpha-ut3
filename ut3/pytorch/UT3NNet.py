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
        #self.conv = SepConv2d(in_channels, growth)
        self.conv = nn.Conv2d(in_channels, growth, 3, 1, 1)
        self.norm = nn.BatchNorm2d(growth)
        self.nlin = nn.ReLU()

    def forward(self, x):
        return torch.cat((x, self.nlin(self.norm(self.conv(x)))), dim=1)

class UT3NNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.board_c = game.getBoardChannels()
        self.action_size = game.getActionSize()
        self.args = args

        super(UT3NNet, self).__init__()
        self.drop = nn.Dropout(self.args.dropout)

        self.conv1 = DenseConv2d(self.board_c+0*self.args.growth, self.args.growth)
        self.conv2 = DenseConv2d(self.board_c+1*self.args.growth, self.args.growth)
        self.conv3 = DenseConv2d(self.board_c+2*self.args.growth, self.args.growth)
        self.conv4 = DenseConv2d(self.board_c+3*self.args.growth, self.args.growth)
        self.conv5 = DenseConv2d(self.board_c+4*self.args.growth, self.args.growth)
        self.conv6 = DenseConv2d(self.board_c+5*self.args.growth, self.args.growth)
        self.conv7 = DenseConv2d(self.board_c+6*self.args.growth, self.args.growth)
        self.conv8 = DenseConv2d(self.board_c+7*self.args.growth, self.args.growth)

        self.out_pi = SepConv2d(self.board_c+8*self.args.growth, 1)
        self.out_v = SepConv2d(self.board_c+8*self.args.growth, 1,
            kernel_size=(self.board_x, self.board_y), padding=0)

    def forward(self, s):

        s = self.conv1(s)
        s = self.conv2(s)
        s = self.conv3(s)
        s = self.conv4(s)
        s = self.conv5(s)
        s = self.conv6(s)
        s = self.conv7(s)
        s = self.conv8(s)

        s = self.drop(s)

        pi = self.out_pi(s).view(-1, self.board_x*self.board_y)
        v = self.out_v(s).view(-1, 1)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
