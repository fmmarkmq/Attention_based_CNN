import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.ABC_Layer import ABC_2D_Agnostic, ABC_2D_Specific, ABC_2D_Large
from model.other_layers import RowWiseLinear, PositionalEncoding, MLP

class Linear_Module(nn.Module):
    def __init__(self, in_features, out_features, in_dim=None, out_dim=None, unflatten=None, activation=None, ltype='linear'):
        super().__init__()
        self.in_dim = in_dim
        if type(self.in_dim) is int:
            self.in_dim = (self.in_dim,)
        self.out_dim = out_dim
        self.unflatten = unflatten

        if activation == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activate = nn.ELU(inplace=True)
        else:
            self.activate = None
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        if self.in_dim is not None:
            self.in_dim = tuple(dim if dim != -1 else len(x.shape)-1 for dim in self.in_dim)
            x = x.permute(*tuple(i for i in range(len(x.shape)) if i not in self.in_dim), *self.in_dim)
            x = x.flatten(-len(self.in_dim), -1)
        x = self.linear(x)
        if self.out_dim is not None:
            out_permute_index = list(range(len(x.shape)-1))
            out_permute_index.insert(self.out_dim, len(x.shape)-1)
            x = x.permute(tuple(out_permute_index))
        if self.unflatten is not None:
            x = x.unflatten(self.out_dim, self.unflatten)
        if self.activate is not None:
            x = self.activate(x)
        return x
    
class Conv_Module(nn.Module):
    def __init__(self, layer_name, layer_paras, length=1, ds_size=(1,1), activation='relu', if_residual=True, hash=None):
        super().__init__()
        self.name = layer_name
        self.input_channel = layer_paras[0]
        self.output_channel = layer_paras[1]
        self.kernel_size = layer_paras[2]
        self.if_residual = if_residual
        self.hash = hash
        if hash is not None:
            self.new_hash = hash.reshape(hash.shape[0], int(hash.shape[1]/ds_size[0]), ds_size[0], int(hash.shape[2]/ds_size[1]), ds_size[1], int(hash.shape[1]/ds_size[0]), ds_size[0], int(hash.shape[2]/ds_size[1]), ds_size[1]).permute(0,1,3,5,7,2,4,6,8).flatten(-4).mean(-1).flatten(-2)
        else:
            self.new_hash = None
  
        if layer_name=='specific':
            self.conv = ABC_2D_Specific
            self.first_conv_paras = (self.input_channel, self.output_channel, self.kernel_size, self.hash, *layer_paras[3:])
            self.mid_conv_paras = (self.output_channel, self.output_channel, self.kernel_size, self.hash, *layer_paras[3:])
            self.last_conv_paras = (self.output_channel, self.output_channel, self.kernel_size, self.hash, *layer_paras[3:])

        elif layer_name=='agnostic':
            self.conv = ABC_2D_Agnostic
            self.first_conv_paras = (self.input_channel, self.output_channel, self.kernel_size, self.hash, *layer_paras[3:])
            self.mid_conv_paras = (self.output_channel, self.output_channel, self.kernel_size, self.hash, *layer_paras[3:])
            self.last_conv_paras = (self.output_channel, self.output_channel, self.kernel_size, self.hash, *layer_paras[3:])
        elif layer_name=='large':
            self.conv = ABC_2D_Large
            self.first_conv_paras = (self.input_channel, self.output_channel, self.kernel_size, layer_paras[3], self.hash, ds_size, *layer_paras[5:])
            self.mid_conv_paras = (self.output_channel, self.output_channel, self.kernel_size, layer_paras[3], self.hash, *layer_paras[4:])
            self.last_conv_paras  = (self.output_channel, self.output_channel, self.kernel_size, layer_paras[3], self.hash, *layer_paras[5:])
        elif layer_name=='cnn2d':
            self.conv = nn.Conv2d
            self.first_conv_paras = (self.input_channel, self.output_channel, self.kernel_size, ds_size, *layer_paras[4:])
            self.mid_conv_paras = (self.output_channel, self.output_channel, self.kernel_size, *layer_paras[3:])
            self.last_conv_paras  = (self.output_channel, self.output_channel, self.kernel_size, *layer_paras[3:])
        
        if activation == 'relu':
            self.activate = nn.ReLU
        elif activation == 'elu':
            self.activate = nn.ELU
        
        self.first_stage = nn.Sequential(self.conv(*self.first_conv_paras), nn.BatchNorm2d(self.output_channel))
        self.mid_stages = nn.ModuleList([])
        for _ in range(length-2):
            self.mid_stages.append(nn.Sequential(self.activate(inplace=True), self.conv(*self.mid_conv_paras), nn.BatchNorm2d(self.output_channel)))
        self.last_stage = nn.Sequential()
        if length > 1 :
            self.last_stage = nn.Sequential(nn.Sequential(self.activate(inplace=True), self.conv(*self.last_conv_paras), nn.BatchNorm2d(self.output_channel)))
        if self.name in ['specific', 'agnostic']:
            self.last_stage.append(nn.AvgPool2d(ds_size))

        if self.if_residual:
            self.input_connect = nn.Sequential()
            if self.input_channel != self.output_channel or ds_size != (1,1):
                self.input_connect.append(nn.Conv2d(self.input_channel, self.output_channel, kernel_size=1, stride=ds_size))
                self.input_connect.append(nn.BatchNorm2d(self.output_channel))
            
        self.final_activate = self.activate(inplace=True)
        
    def forward(self, inputs):
        x = inputs
        x = self.first_stage(x)
        for mid_stage in self.mid_stages:
            x = mid_stage(x)
        x = self.last_stage(x)
        if self.if_residual:
            x = x + self.input_connect(inputs)
        x = self.final_activate(x)
        return x

class Attention_Module(nn.Module):
    def __init__(self, d_model, n_head, length=1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.n_head = n_head
        self.length = length
        self.trans_modules = nn.ModuleList([])
        for i in range(self.length):
            self.trans_modules.append(nn.TransformerEncoderLayer(self.d_model, self.n_head, 
                                                                 batch_first=True, norm_first=True, 
                                                                 dim_feedforward=2*self.d_model))
        self.position_embedding = nn.Embedding(1000, self.d_model)
        self.primer = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.uniform_(self.primer, a=-np.sqrt(1/self.d_model), b=np.sqrt(1/self.d_model))
    
    def add_position_embedding(self, x):
        B, N, D = x.shape
        positions = torch.arange(N).reshape(1,N).repeat(B,1).long().to(self.device)
        x = x + self.position_embedding(positions)
        return x
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(-2).transpose(1,2)
        primers = self.primer.repeat(B, 1, 1)
        x = torch.concat([primers, x], dim=1)
        x = self.add_position_embedding(x)
        for trans_encoder in self.trans_modules:
            x = trans_encoder(x)
        x = x[:, 0, :].reshape(B, self.d_model, 1, 1)
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
            if layer_name in ['specific', 'agnostic', 'large', 'cnn2d', 'cnn1d']:
                modules.append(Conv_Module(layer_name, *paras, hash=hash))
                hash = modules[-1].new_hash
            elif layer_name in 'attention':
                modules.append(Attention_Module(*paras))
            elif layer_name=='linear':
                modules.append(Linear_Module(*paras))
            elif layer_name=='softmax':
                modules.append(nn.Softmax(paras))
            elif layer_name=='maxpool':
                modules.append(nn.MaxPool2d(*paras))
            elif layer_name=='adptavgpool':
                modules.append(nn.AdaptiveAvgPool2d(paras))
        return modules