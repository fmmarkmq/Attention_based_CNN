import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
from model.ABC_Layer import ABC_2D

# class RowWiseLinear(nn.Module):
#     def __init__(self, height, width):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.weights = nn.Parameter(torch.ones(height, 1, width))
#         self.register_parameter('weights', self.weights)
#         # self.weights = nn.Parameter(weights)
#         # self.weights = torch.ones(height, 1, width).to('cuda')
#         # self.register_buffer('mybuffer', self.weights)

        
#     def forward(self, x):
#         x_unsqueezed = x.unsqueeze(-1)
#         w_times_x = torch.matmul(self.weights, x_unsqueezed)
#         return w_times_x.squeeze()


class ABC_Net(nn.Module):
    def __init__(self, args, hash):
        super(ABC_Net, self).__init__()
        self.args = args
        self.hash = hash

        self.H = self.args.input_height
        self.W = self.args.input_width
        self.pixel_number = self.H*self.W

       
        # self.ABC_2D = ABC_2D(in_channel=1,
        #                   kernel_size=9,
        #                   pixel_number=784,
        #                   kernel_number_per_pixel=6,
        #                   hash=self.hash)
        

        self.ABC_2D = ABC_2D(in_channel=self.args.input_channel,
                          kernel_size=self.args.kernel_size,
                          pixel_number=self.pixel_number,
                          kernel_number_per_pixel=self.args.knpp,
                          hash=self.hash)

        self.fc1 = nn.Linear(self.args.knpp*self.pixel_number, self.predict_len*self.pixel_number)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # start_time = time.time
        B,C,H,W = x.shape
        x = self.ABC_2D(x)
        x = self.relu(x)
        # B, kernel_number_per_pixel, H*W

        x = x.reshape(B, -1)
        x = self.fc1(x)
        # B, 4*5200
        x = x.reshape(B, self.args.predict_len, H, W)
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