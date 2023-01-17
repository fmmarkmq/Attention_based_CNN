import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.ABC_Layer import ABC_2D_Agnostic, ABC_2D_Specific, ABC_2D_Large

class Linear_Module(nn.Module):
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
    
class Conv_Module(nn.Module):
    def __init__(self, layer_name, paras, hash=None):
        super().__init__()
        self.name = layer_name
        if hash is not None:
            para_list = list(paras)
            para_list.insert(4, hash)
            self.paras = tuple(para_list)
        else:
            self.paras = paras
        self.hash = hash
        if layer_name=='specific':
            self.conv = ABC_2D_Specific(*self.paras)
            self.maxpool = nn.MaxPool2d(paras[-1])
        elif layer_name=='agnostic':
            self.conv = ABC_2D_Agnostic(*self.paras)
            self.maxpool = nn.MaxPool2d(paras[-1])
        elif layer_name=='large':
            self.conv = ABC_2D_Large(*self.paras)
            self.maxpool = nn.MaxPool2d(paras[-1])
        elif layer_name=='cnn2d':
            self.conv = nn.Conv2d(*self.paras)
            self.maxpool = nn.MaxPool2d(paras[-2])
        elif layer_name=='cnn1d':
            self.conv = nn.Conv1d(*self.paras)
            self.maxpool = nn.MaxPool2d(paras[-2])
        self.fc = nn.Linear(self.paras[0], self.paras[1])
        self.norm = nn.BatchNorm2d(self.paras[1])
        self.activate = nn.ReLU()
        
    
    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.norm(x)
        x = x + self.fc(self.maxpool(inputs).transpose(1,3)).transpose(1,3)
        x = self.activate(x)
        return x

class Attention_Module(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.n_head = n_head
        self.trans_encoder = nn.TransformerEncoderLayer(self.d_model+1, self.n_head, 
                                                        batch_first=True, norm_first=True, 
                                                        dim_feedforward=2*(self.d_model+1))
        self.primer = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.uniform_(self.primer, a=-np.sqrt(1/self.d_model), b=np.sqrt(1/self.d_model))
    
    def add_position_encoding(self, x):
        B, C, HW = x.shape
        positions = torch.arange(C).reshape(1,C,1).repeat(B,1,1).to(self.device)
        x = torch.concat([positions,x],dim=-1)
        return x
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(-2).transpose(1,2)
        primers = self.primer.repeat(B, 1, 1)
        x = torch.concat([primers, x], dim=1)
        x = self.add_position_encoding(x)
        x = self.trans_encoder(x)
        x = x[:, 0, :].reshape(B, self.d_model+1, 1, 1)
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
            if layer_name in ['specific', 'agnostic', 'large']:
                module = Conv_Module(layer_name, paras, hash)
                modules.append(module)
                hash = module.conv.new_hash
            elif layer_name in ['cnn2d', 'cnn1d']:
                modules.append(Conv_Module(layer_name, paras))
            elif layer_name in 'attention':
                modules.append(Attention_Module(*paras))
            elif layer_name=='linear':
                modules.append(Linear_Module(*paras))
            elif layer_name=='softmax':
                modules.append(nn.Softmax(paras))
        return modules