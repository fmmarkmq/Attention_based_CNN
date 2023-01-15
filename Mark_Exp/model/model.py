import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
from model.ABC_Layer import ABC_2D_Agnostic, ABC_2D_Specific, ABC_2D_Large

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

class Linear_with_process(nn.Module):
    def __init__(self, in_features, out_features, in_dim=None, out_dim=None, unflatten=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.unflatten = unflatten
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        if self.in_dim is not None:
            x = x.permute(*tuple(i for i in range(len(x.shape)) if i not in self.in_dim), *self.in_dim)
            x = x.flatten(-len(self.in_dim), -1)
        x = self.fc(x)
        if self.out_dim is not None:
            out_permute_index = list(range(len(x.shape)-1))
            out_permute_index.insert(self.out_dim, len(x.shape)-1)
            x = x.permute(tuple(out_permute_index))
        if self.unflatten is not None:
            x = x.unflatten(self.out_dim, self.unflatten)
        return x


class ABC_Net(nn.Module):
    def __init__(self, args, hash):
        super(ABC_Net, self).__init__()
        self.args = args
        self.hash = hash
        self.full_modules = self._make_modules(self.args.layers, self.hash)
    
    def forward(self,x):
        B,C,H,W = x.shape
        for i, module in enumerate(self.full_modules):
            x = module(x)
        return x

    def _make_modules(self, layers, hash):
        modules = nn.ModuleList([])
        for layer_name, paras in layers:
            if layer_name=='specific':
                modules.append(ABC_2D_Specific(*paras, hash=hash))
                modules.append(nn.ReLU(inplace=True))
            elif layer_name=='agnostic':
                modules.append(ABC_2D_Agnostic(*paras, hash=hash))
                modules.append(nn.ReLU(inplace=True))
            elif layer_name=='large':
                modules.append(ABC_2D_Large(*paras, hash=hash))
                modules.append(nn.ReLU(inplace=True))
            elif layer_name=='cnn2d':
                modules.append(nn.Conv2d(*paras))
                modules.append(nn.ReLU)
            elif layer_name=='cnn1d':
                modules.append(nn.Conv1d(*paras))
                modules.append(nn.ReLU)
            elif layer_name=='linear':
                modules.append(Linear_with_process(*paras))
            elif layer_name=='softmax':
                modules.append(nn.Softmax(paras))
        return modules