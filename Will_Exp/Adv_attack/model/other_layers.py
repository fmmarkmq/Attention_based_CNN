import torch
from torch import nn
import torch.nn.functional as F
from torch import  Tensor
import numpy as np
import math

class MLP(torch.nn.Module): 
    def __init__(self, in_features: list, out_features: list, dims, activation='relu'):
        super(MLP,self).__init__() 
        self.relu = nn.ReLU(inplace=True)
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
