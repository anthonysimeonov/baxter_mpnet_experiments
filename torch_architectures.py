import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


# DMLP Model-Path Generator
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(),
                    nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
                    nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
                    nn.Linear(64, 32), nn.PReLU(),
                    nn.Linear(32, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out


class MLP_Path(nn.Module):
     def __init__(self, input_size, output_size):
         super(MLP_Path, self).__init__()
         self.fc = nn.Sequential(
                    nn.Linear(input_size, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, output_size))

     def forward(self, x):
         out = self.fc(x)
         return out


class VoxelEncoder(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size, output_size):
        super(VoxelEncoder, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=[5,5,5], stride=[2,2,2]),
            nn.PReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.PReLU(),
            nn.MaxPool3d(kernel_size=[2,2,2], stride=[2,2,2])
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class VoxelEncoder2(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size, output_size):
        super(VoxelEncoder2, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=[5,5,5], stride=[2,2,2]),
            nn.PReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.PReLU()
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 256),
            nn.PReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class VoxelEncoder3(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size, output_size):
        super(VoxelEncoder3, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.PReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=[5,5,5], stride=[2,2,2]),
            nn.PReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=[5,5,5], stride=[2,2,2]),
            nn.PReLU()
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class MultiViewVoxelEncoder3(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size, output_size):
        super(MultiViewVoxelEncoder3, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=8, kernel_size=[3,3], stride=[1,1]),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[5,5], stride=[2,2]),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[5,5], stride=[2,2]),
            nn.PReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=8, kernel_size=[3,3], stride=[1,1]),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[5,5], stride=[2,2]),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[5,5], stride=[2,2]),
            nn.PReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=8, kernel_size=[3,3], stride=[1,1]),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[5,5], stride=[2,2]),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[5,5], stride=[2,2]),
            nn.PReLU()
        )
        x = self.encoder1(torch.autograd.Variable(torch.rand([1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        print('length of the output of one encoder')
        print(first_fc_in_features)
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features*3, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        # x shape: BxCxWxHxD
        size = x.size()
        print('input size:')
        print(size)
        x1 = x.permute(0, 1, 4, 2, 3).reshape(size[0], -1, size[2], size[3])# transpose to Bx(CxD)xWxH
        x2 = x.permute(0, 1, 3, 2, 4).reshape(size[0], -1, size[2], size[4])# transpose to Bx(CxH)xWxD
        x3 = x.permute(0, 1, 2, 3, 4).reshape(size[0], -1, size[3], size[4])# transpose to Bx(CxW)xHxD
        print('x1 size:')
        print(x1.size())
        print('x2 size:')
        print(x2.size())
        print('x3 size:')
        print(x3.size())
        x1, x2, x3 = self.encoder1(x1),self.encoder2(x2),self.encoder3(x3)
        print('after encoder x1 size:')
        print(x1.size())
        x1, x2, x3 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1)
        # cat x1 x2 x3 into x
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.head(x)
        return x





class VoxelEncoder4(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size, output_size):
        super(VoxelEncoder4, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.PReLU())
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, output_size)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class VoxelEncoder5(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size, output_size):
        super(VoxelEncoder5, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=[7,7,7], stride=[2,2,2]),
            nn.PReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[5,5,5], stride=[2,2,2]),
            nn.PReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.PReLU()
            #nn.MaxPool3d(kernel_size=[2,2,2], stride=[2,2,2])
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x





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
