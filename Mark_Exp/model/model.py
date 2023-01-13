import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
from model.ABC_Layer import ABC_2D_Agnotic, ABC_2D_Specific, ABC_2D_Large

class RowWiseLinear(nn.Module):
    def __init__(self, height, width, out_width):
        super().__init__()
        self.height = height
        self.width = width
        self.weights = nn.Parameter(torch.ones(height, out_width, width))
        self.register_parameter('weights', self.weights)
        # self.weights = nn.Parameter(weights)
        # self.weights = torch.ones(height, 1, width).to('cuda')
        # self.register_buffer('mybuffer', self.weights)

        
    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)
        w_times_x = torch.matmul(self.weights, x_unsqueezed)
        return w_times_x


class ABC_Net(nn.Module):
    def __init__(self, args, hash):
        super(ABC_Net, self).__init__()
        self.args = args
        self.hash = hash
        self.pixel_number = self.args.input_height * self.args.input_width

        self.ABC_2D = ABC_2D_Specific(in_channel=self.args.input_channel,
                          kernel_size=self.args.kernel_size,
                          pixel_number=self.pixel_number,
                          kernel_number_per_pixel=self.args.knpp,
                          hash=self.hash)
        
        self.ABC_2D_1 = ABC_2D_Agnotic(in_channel=self.args.knpp,
                          kernel_size=self.args.kernel_size,
                          pixel_number=self.pixel_number,
                          kernel_number_per_pixel=self.args.knpp2,
                          hash=self.hash)

        # self.ABC_2D = ABC_2D_Large(in_channel=self.args.input_channel,
        #                   kernel_size=(5,5),
        #                   perceptual_size=self.args.kernel_size,
        #                   out_channel=self.args.knpp,
        #                   hash=self.hash)
        # self.ABC_2D_1 = ABC_2D_Large(in_channel=self.args.knpp,
        #                   kernel_size=(5,5),
        #                   perceptual_size=self.args.kernel_size,
        #                   out_channel=self.args.knpp2,
        #                   hash=self.hash)
        
        if self.args.name == "mnist":
            self.fc1 = nn.Linear(self.args.knpp2*self.args.input_height*self.args.input_width, 10)
        elif self.args.name == "atd":
            # self.rwl = RowWiseLinear(5200, self.args.knpp, out_width=self.args.predict_len)
            # self.fc1 = nn.Linear(self.args.knpp2, self.args.predict_len)
            # self.fc2 = nn.Linear(self.args.input_height*self.args.input_width, self.args.input_height*self.args.input_width)
            self.fc1 = nn.Linear(self.args.knpp2, 1)
            self.fc2 = nn.Linear(self.args.input_height, self.args.predict_len)
        
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(1)
    
    def forward(self,x):
        if self.args.name == "mnist":
            return self.mnist_forward(x)
        elif self.args.name == "atd":
            return self.atd_forward(x)

    # atd_model
    def atd_forward(self, x):
        B,C,H,W = x.shape
        x = self.ABC_2D(x)
        x = self.relu(x)
        # B, -1, H, W

        x = self.ABC_2D_1(x)
        x = self.relu(x)
        # B, -1, H, W
        x = x.transpose(1,3)
        x = self.fc1(x)
        # x = x.transpose(1,3)
        x = x.permute(0,3,1,2)
        x = self.fc2(x) 
        x = x.permute(0,3,1,2)
        x = x.reshape(B, 1, self.args.predict_len, W)
        # B, 4, 1, 5200
        return x

    # mnist_model
    def mnist_forward(self, x):
        B,C,H,W = x.shape
        x = self.ABC_2D(x)
        x = self.relu(x)
        # B, -1, H, W

        x = self.ABC_2D_1(x)
        x = self.relu(x)
        # B, -1, H, W
        x = x.reshape(B, -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class CNN_Net(nn.Module):
    def __init__(self, args):
        super(CNN_Net, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(
            in_channels = self.args.input_channel,
            out_channels = self.args.knpp,
            kernel_size = 3,
            stride = 1,
            padding= 1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.args.knpp,
            out_channels = self.args.knpp2,
            kernel_size=3,
            stride = 1,
            padding= 1,
            bias=False
        )
        self.fc1 = nn.Linear(self.args.knpp2*28*28, 10)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        # start_time = time.time
        B,C,H,W = x.shape
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(B, -1)

        x = self.fc1(x)
        x = self.softmax(x)
        return x