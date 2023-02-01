import torch
from torch import nn
import torch.nn.functional as F
from torch import  Tensor
import numpy as np
import math


class PowerExpansion(nn.Module):
    def __init__(self, out_powers: int, if_learnable=False):
        super(PowerExpansion, self).__init__()
        self.out_powers = out_powers
        self.if_learnable = if_learnable
        if if_learnable:
            self.weights = nn.Parameter(torch.arange(1, out_powers+1).to(torch.float32).reshape(1,out_powers))
        else:
            self.weights = torch.arange(1, out_powers+1).to(torch.float32).reshape(1,out_powers)
    
    def forward(self, x):
        assert x.min() >= 0
        x = x.log()
        x = x.unsqueeze(-1)
        x = torch.matmul(x, self.weights)
        x = x.exp()
        return x


class PowerLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dim: int, input_min: torch.float32, eps=1e-05):
        super(PowerLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.input_min = input_min
        self.eps = eps
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        x_sign = x.sign()
        x = x.abs().log()
        x = self.linear(x.transpose(self.dim, -1)).transpose(self.dim, -1)
        x = x.exp()*x_sign
        return x
        

class MLP(nn.Module): 
    def __init__(self, in_features: list, out_features: list, dims, activation='relu'):
        super(MLP,self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.test_paras(in_features, out_features, dims, activation)

        if type(dims) is int:
            self.dims = [dims]*(len(in_features))
        
        if activation == 'relu':
            self.activate = nn.ReLU()
        elif activation == 'elu':
            self.activate = nn.ELU()
        
        self.fcs = nn.ModuleList([])
        for i in range(len(self.in_features)):
            self.fcs.append(nn.Linear(in_features=in_features[i], out_features=out_features[i]))
    
    def forward(self,x):
        for i, fc in enumerate(self.fcs):
            x = x.transpose(self.dims[i], -1)
            x = fc(x)
            x = x.transpose(self.dims[i], -1)
            if i != len(self.fcs)-1:
                fc = self.activate(x)
        return x
    
    def test_paras(self, in_features, out_features, dims, activation):
        assert type(in_features) is list
        assert type(out_features) is list
        assert len(in_features) == len(out_features)
        assert len(in_features) >= 2
        assert (type(dims) is int) or (len(dims) == len(in_features))
        assert activation in ['relu', 'elu']

class RowWiseLinear(nn.Module):
    def __init__(self, height, width, out_width):
        super().__init__()
        self.height = height
        self.width = width
        self.weights = nn.Parameter(torch.empty(height, out_width, width))
        nn.init.uniform_(self.weights, a=-np.sqrt(1/out_width), b=np.sqrt(1/out_width))

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)
        w_times_x = torch.matmul(self.weights, x_unsqueezed)
        return w_times_x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)