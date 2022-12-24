import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader


class ABC_Net(nn.Module):
    def __init__(self):
        super(ABC_Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 3,
            stride = 1,
            # padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels = 32,
            kernel_size=3,
            stride=1,
            # padding=1
        )

        self.fc1 = nn.Linear(32*24*24, 10)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(B, -1)

        x = self.fc1(x)
        # print(x.shape)
        return x