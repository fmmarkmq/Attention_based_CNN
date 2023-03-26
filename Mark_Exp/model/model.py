import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.ABC_Layer import ABC_2D_Agnostic, ABC_2D_Specific, ABC_2D_Large
from model.ABC_Layer import NeighborAttention, AttentionConv, AttentionResConv, DepthAttentionResConv
from model.other_layers import PowerExpansion, MLP, ResMLP, ResLinear
from model.conv_layers import DERIConv2d, WSConv2d, WSAConv2d
from utils.tools import dotdict

class Linear_Module(nn.Module):
    def __init__(self, in_features, out_features, in_dim=None, out_dim=None, unflatten=None, activation=None, device=None):
        super().__init__()
        self.in_dim = in_dim
        if type(self.in_dim) is int:
            self.in_dim = (self.in_dim,)
        self.out_dim = out_dim
        self.unflatten = unflatten

        # self.norm = nn.BatchNorm1d(out_features)

        if activation == 'relu':
            self.activate = nn.ReLU()
        elif activation == 'elu':
            self.activate = nn.ELU()
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
        # x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class Conv_Module(nn.Module):
    def __init__(self, layer_name, layer_paras, length=1, ds_position=None, ds_size=None, activation='relu', if_residual=True, norm='batch', hash=None):
        super().__init__()
        self.name = layer_name
        self.layer_paras = layer_paras
        self.input_channel = layer_paras[0]
        self.output_channel = layer_paras[1]
        self.length = length
        self.ds_position = ds_position
        if ds_size is None:
            ds_size = (1,1)
        self.ds_size = ds_size
        self.if_residual = if_residual
        self.hash = hash
        self.new_hash = self._build_new_hash()
        self.conv_paras = self._make_conv_paras()

        if layer_name=='specific':
            self.conv = ABC_2D_Specific
        elif layer_name=='agnostic':
            self.conv = ABC_2D_Agnostic
        elif layer_name=='large':
            self.conv = ABC_2D_Large
        elif layer_name=='cnn2d':
            self.conv = nn.Conv2d
        elif layer_name=='dericonv2d':
            self.conv = DERIConv2d
        elif layer_name=='nba2d':
            self.conv = NeighborAttention
        elif layer_name== 'atc2d':
            self.conv = AttentionConv
        elif layer_name== 'atrc2d':
            self.conv = AttentionResConv
        elif layer_name== 'datrc2d':
            self.conv = DepthAttentionResConv
        elif layer_name== 'wsc2d':
            self.conv = WSConv2d
        elif layer_name== 'wsca2d':
            self.conv = WSAConv2d    

        if activation == 'relu':
            self.activate = nn.ReLU
        elif activation == 'elu':
            self.activate = nn.ELU
        elif activation == False:
            self.activate = nn.Identity

        self.pool = nn.AvgPool2d
        # self.pool = nn.MaxPool2d
            

        layerlist = []
        for i, conv_para in enumerate(self.conv_paras):
            layerlist.append(self.conv(*conv_para, bias=False))
            layerlist.append(nn.BatchNorm2d(self.output_channel))
            if i < self.length-1:
                layerlist.append(self.activate())
        if (self.name in ['specific', 'agnostic']) and (ds_size not in [(1,1), None]):
            layerlist.append(self.pool(ds_size))
        if (self.name in ['nba2d', 'atc2d', 'atrc2d', 'datrc2d']) and (ds_size not in [(1,1), None]):
            if ds_position == 'first':
                layerlist.insert(1, self.pool(ds_size))
            elif ds_position == 'last':
                layerlist.insert(-2, self.pool(ds_size))
        
        self.layerlist = nn.Sequential(*layerlist)

        if self.if_residual:
            if self.input_channel != self.output_channel or (ds_size not in [(1,1), None]):
                self.input_connect = nn.Sequential(nn.Conv2d(self.input_channel, self.output_channel, kernel_size=1, stride=ds_size),
                                                    nn.BatchNorm2d(self.output_channel))g
            else:
                self.input_connect = nn.Identity()
            
        self.final_activate = self.activate()
        self._weight_initialize()
        
    def forward(self, inputs):
        x = inputs
        x = self.layerlist(x)
        if self.if_residual:
            x += self.input_connect(inputs)
        x = self.final_activate(x)
        return x

    def _make_conv_paras(self):
        layer_name = self.name
        layer_paras = self.layer_paras

        layer_paras = (self.output_channel, *layer_paras[1:])
        if layer_name in ['specific','agnostic']:
            layer_paras = (*layer_paras[:3], self.hash, *layer_paras[3:])
        elif layer_name=='large':
            layer_paras = (*layer_paras[:4], self.hash, *layer_paras[4:])
        
        conv_paras = [layer_paras]* self.length
        conv_paras[0] = (self.input_channel, *conv_paras[0][1:])

        assert self.ds_position in ['first', 'last', None]
        if self.ds_position is not None:
            if self.ds_position == 'first':
                ds_index = 0
            elif self.ds_position == 'last':
                ds_index = -1
            if layer_name=='large':
                conv_paras[ds_index] = (*conv_paras[ds_index][:5], self.ds_size, *conv_paras[ds_index][6:])
            elif layer_name in ['cnn2d', 'dericonv2d', 'wsc2d', 'wsac2d']:
                conv_paras[ds_index] = (*conv_paras[ds_index][:3], self.ds_size, *conv_paras[ds_index][4:])
        return conv_paras
    
    def _weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.layerlist[-1].weight, 1)
    
    def _build_new_hash(self):
        if (self.hash is not None) and (self.ds_size != (1,1)) and (self.ds_size is not None):
            HC, HH, HW, HHW = self.hash.shape
            DH, DW = self.ds_size
            new_hash = self.hash.reshape(HC, int(HH/DH), DH, int(HW/DW), DW, int(HH/DH), DH, int(HW/DW), DW)
            new_hash = new_hash.permute(0,1,3,5,7,2,4,6,8).flatten(-4).mean(-1).flatten(-2)
        else:
            new_hash = self.hash
        return new_hash


class Attention_Module(nn.Module):
    def __init__(self, d_model, n_head, length=1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.n_head = n_head
        self.length = length
        self.trans_modules = nn.ModuleList([])
        for _ in range(self.length):
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
    def __init__(self, args: dotdict, hash):
        super(ABC_Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.hash = hash
        self.full_modules = self._make_modules(self.args.layers, self.hash)
        
    
    def forward(self, x):
        for _, module in enumerate(self.full_modules):
            x = module(x)
        return x

    def _make_modules(self, layers, hash):
        modules = nn.ModuleList([])
        for layer_name, paras in layers:
            if layer_name in ['specific', 'agnostic', 'large', 'cnn2d', 'dericonv2d',
                              'nba2d', 'atc2d', 'atrc2d', 'datrc2d', 'wsc2d', 'wsac2d']:
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
            elif layer_name=='avgpool':
                modules.append(nn.AvgPool2d(*paras))
            elif layer_name=='adptavgpool':
                modules.append(nn.AdaptiveAvgPool2d(paras))
            elif layer_name=='powerexpansion':
                modules.append(PowerExpansion(*paras))
        return modules
    

class PipelineParallelABC_Net(nn.Module):
    def __init__(self, args: dotdict, hash, devices, split_size=20):
        super(PipelineParallelABC_Net, self).__init__()
        self.args = args
        self.hash = hash
        self.device = devices
        self.split_size = split_size

        self.args_first_half, self.args_second_half = self._build_args()
        self.module_first_half = ABC_Net(args=self.args_first_half, hash=self.hash).to(self.device[0])
        self.module_second_half = ABC_Net(args=self.args_second_half, hash=self.module_first_half.full_modules[-1].new_hash).to(self.device[1])


    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.module_first_half(s_next).to(self.device[1])
        ret = []

        for s_next in splits:
            s_prev = self.module_second_half(s_prev)
            ret.append(s_prev)
            s_prev = self.module_first_half(s_next).to(self.device[1])

        s_prev = self.module_second_half(s_prev)
        ret.append(s_prev)

        return torch.concat(ret, dim=0).to(self.device[0])


    def _build_args(self):
        # full_modules_len = len(self.args.layers)
        args_first_half = dotdict(self.args.copy())
        args_second_half = dotdict(self.args.copy())
        args_first_half.layers = self.args.layers[:3]
        args_second_half.layers = self.args.layers[3:]
        return args_first_half, args_second_half
    

class ResMLPNet(nn.Module):
    def __init__(self):
        super(ResMLPNet, self).__init__()
        self.res_mlp = ResMLP(1, [784], [784])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(784, 10)
    
    def forward(self,x):
        B,C,H,W = x.shape
        x = x.reshape(B,-1)
        x = self.res_mlp(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )