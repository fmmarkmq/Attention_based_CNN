import torch
from torch import nn
import torch.nn.functional as F
from torch import  Tensor
import numpy as np
import math
from torch.nn import Linear


class PowerExpansion(nn.Module):
    def __init__(self, out_powers: int, in_channel, power_learnable=False, fc=True, device=None):
        super(PowerExpansion, self).__init__()
        self.out_powers = out_powers
        self.in_channel = in_channel
        self.learnable = power_learnable
        self.fc = fc
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if power_learnable:
            self.expansion = nn.Parameter(torch.arange(1, out_powers+1, dtype=torch.float32).reshape(1,out_powers))
        else:
            self.expansion = torch.arange(1, out_powers+1, dtype=torch.float32, device=self.device).reshape(1,out_powers)
        if self.fc:
            self.linear = RowWiseLinear(in_channel, out_powers, 1)
            # self.linear = nn.Linear(out_powers, 1)
            
    
    def forward(self, x):
        B,C,H,W = x.shape
        if x.min()<0:
            print(x.min())
        assert x.min() >= 0
        x = x.log()
        x = x.unsqueeze(-1)
        x = torch.matmul(x, self.expansion)
        x = x.exp()
        if self.fc:
            x = x.transpose(1,3)
            x = self.linear(x).squeeze(-1)
            x = x.transpose(1,3)
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
        super(MLP, self).__init__() 
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
        super(RowWiseLinear, self).__init__()
        self.height = height
        self.width = width
        self.weights = nn.Parameter(torch.empty(height, out_width, width))
        nn.init.uniform_(self.weights, a=-np.sqrt(1/out_width), b=np.sqrt(1/out_width))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = torch.matmul(self.weights, x)
        return x.squeeze(-1)


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


class ResLinear(nn.Linear):
    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super(ResLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.n = n
        if n > 0:
            self.fc1 = ResLinear(n-1, in_features, in_features, bias, device, dtype)
            self.fc2 = ResLinear(n-1, in_features, out_features, bias, device, dtype)
    
    def forward(self, input):
        x = F.linear(input, self.weight, self.bias)
        if self.n>0:
            w1 = self.fc1(input).unsqueeze(-2)
            w2 = self.fc2(input).unsqueeze(-1)
            w = torch.matmul(w2, w1)
            x += torch.matmul(w, input.unsqueeze(-1)).squeeze(-1)
        return x


class ResMLP(nn.Module): 
    def __init__(self, n: int, in_features: list, out_features: list, dims=-1, activation='relu'):
        super(ResMLP, self).__init__()
        self.n = n
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
            self.fcs.append(ResLinear(self.n, in_features=in_features[i], out_features=out_features[i]))
    
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
        # assert len(in_features) >= 2
        assert (type(dims) is int) or (len(dims) == len(in_features))
        assert activation in ['relu', 'elu']
