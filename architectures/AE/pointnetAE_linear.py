"""
The original pointnet implementation is from
ref: https://github.com/fxia22/pointnet.pytorch
Siamese features is added upon that
Adding Siamese feature notice:
1d conv layers weights are already shared
fc layers need to appy separately, and then apply bn together
"""

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
        self.output_size = output_size
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 256, 1)

        self.global_fc1 = nn.Linear(256, 128)
        self.global_fc2 = nn.Linear(128, 128)

        self.conv6 = torch.nn.Conv1d(256, 256, 1)
        self.conv7 = torch.nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        # x shape: [B*2, shape]
        n_pts = x.size()[2]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        pointfeat = x
        x = F.relu(x)
        # global features
        x = self.conv5(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        x = F.relu(self.global_fc1(x))
        x = self.global_fc2(x)
        x = x.view(-1, 128, 1).repeat(1, 1, n_pts)
        x = torch.cat([x, pointfeat], 1)
        # new local features -> global embedding
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_size)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(len(x), 3, -1)
        return x
