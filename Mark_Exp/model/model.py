import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
from model.ABC_Layer import ABC_2D


class ABC_Net(nn.Module):
    def __init__(self, args, hash):
        super(ABC_Net, self).__init__()
        self.args = args
        self.hash = hash

        self.ABC_2D = ABC_2D(in_channel=1,
                          kernel_size=9,
                          pixel_number=784,
                          kernel_number_per_pixel=10,
                          hash=self.hash)

        self.fc1 = nn.Linear(10*28*28, 10)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        # start_time = time.time
        B,C,H,W = x.shape
        x = self.ABC_2D(x)
        x = self.relu(x)
        # B, kernel_number_per_pixel, H*W

        x = x.reshape(B, -1)
        x = self.fc1(x)
        # B, 10

        x = self.softmax(x)
        return x




# class ABC_Net(nn.Module):
#     def __init__(self, args):
#         super(ABC_Net, self).__init__()
#         self.args = args

#         self.conv1 = nn.Conv2d(
#             in_channels = 1,
#             out_channels = 10,
#             kernel_size = 3,
#             stride = 1,
#             # padding=1
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=10,
#             out_channels = 20,
#             kernel_size=3,
#             stride = 1,
#             # padding=1
#         )
#         self.fc1 = nn.Linear(20*24*24, 10)
#         self.relu = nn.ReLU(inplace=True)
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.softmax = nn.Softmax(1)
    
#     def forward(self, x):
#         # start_time = time.time
#         B,C,H,W = x.shape
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)

#         x = x.view(B, -1)

#         x = self.fc1(x)
#         x = self.softmax(x)
#         return x