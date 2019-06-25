from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 786), nn.PReLU(),
                                     nn.Linear(786, 512), nn.PReLU(),
                                     nn.Linear(512, 256), nn.PReLU(),
                                     nn.Linear(256, output_size))

    def forward(self, x):
        x = self.encoder(x)
        return x


class Encoder_End2End(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder_End2End, self).__init__() #16053 input, 60 output usually
        self.encoder = nn.Sequential(nn.Linear(input_size, 256), nn.PReLU(),
                                     nn.Linear(256, 256), nn.PReLU(),
                                     nn.Linear(256, output_size))

    def forward(self, x):
        x = self.encoder(x)
        return x
