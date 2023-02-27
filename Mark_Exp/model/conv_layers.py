import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, device=None):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.bias = bias
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_len = kernel_size[0]*kernel_size[1]
        self.new_shape = None

    def _new_shape(self, H, W):
        SH,SW = self.stride
        PH, PW = self.padding
        DH, DW = self.dilation
        KH, KW = self.kernel_size
        H_new = (H + 2*PH - DH*(KH-1)-1)//SH + 1
        W_new = (W + 2*PW - DW*(KW-1)-1)//SW + 1
        return H_new, W_new


class DERIConv2d(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None):
        super(DERIConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device)
        self.center_idx = (self.kernel_len-1)//2
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        self.conv = nn.Conv2d(in_channels*self.kernel_len, out_channels, kernel_size=1, groups=groups, bias=bias)


    def forward(self, x:torch.tensor): 
        B,C,H,W = x.shape
        if self.new_shape is None:
            self.new_shape = self._new_shape(H,W)
        x = self.unfold(x).reshape(B, C, self.kernel_len, *self.new_shape)
        x[:,:, self.center_idx] *=2
        x = x - x[:,:, self.center_idx:self.center_idx+1]/2
        x = x.flatten(1,2)
        x = self.conv(x)
        return x
        

class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class WSAConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        weight = self.relu(weight+0.5)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

