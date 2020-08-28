from torch.nn.modules.module import Module
import sys
import os
sys.path.insert(0,'/home/badri/bns332/foldingnet/nndistance/functions')
import nnd
from functions.nnd import NNDFunction

class NNDModule(Module):
    def forward(self, input1, input2):
        return NNDFunction()(input1, input2)
