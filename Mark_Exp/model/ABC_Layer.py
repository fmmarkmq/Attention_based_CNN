import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.conv_layers import Conv


class AttentionConv(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=4, bias=True, device=None):
        super(AttentionConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device)

        self.center_idx = (self.kernel_len-1)//2
        self.group_out_channels = self.kernel_len * in_channels // groups
        
        self.conv_query = nn.Conv2d(in_channels, in_channels*self.kernel_len, kernel_size, padding='same', groups=in_channels, bias=bias)
        self.conv_key = nn.Conv2d(in_channels, in_channels*self.kernel_len, kernel_size, padding='same', groups=in_channels, bias=bias)
        self.conv_value = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=bias)
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        
        self.softmax = nn.Softmax(-1)
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.new_shape is None:
            self.new_shape = self._new_shape(H,W)
            self.new_pixel = self.new_shape[0]*self.new_shape[1]
        query = self.conv_query(x)
        key = self.conv_key(x)
        value = self.conv_value(x)
        value = self.norm(value)
        value = self.relu(value)

        query = self.unfold(query)
        key = self.unfold(key)
        value = self.unfold(value)

        query = query.reshape(B, self.groups, self.group_out_channels, self.kernel_len, self.new_pixel).transpose(2,4)
        query = query[:,:,:,self.center_idx:self.center_idx+1]

        key = key.reshape(B, self.groups, self.group_out_channels, self.kernel_len, self.new_pixel).transpose(2,4)
        key = key.transpose(3,4)

        value = value.reshape(B, self.groups, self.out_channels//self.groups, self.kernel_len, self.new_pixel).transpose(2,4)

        out = torch.matmul(query, key)
        out = self.softmax(out)
        out = torch.matmul(out, value)
        out = out.transpose(2,4).reshape(B, self.out_channels, *self.new_shape)
        return out

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_query.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_key.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_value.weight, mode='fan_out', nonlinearity='relu')
    

# class AttentionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=4, bias=True):
#         super(AttentionConv, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         if isinstance(kernel_size, int):
#             self.kernel_size = (kernel_size, kernel_size)
#         self.stride = stride
#         if isinstance(stride, int):
#             self.stride = (stride, stride)
#         self.padding = padding
#         if isinstance(padding, int):
#             self.padding = (padding, padding)
#         self.dilation = dilation
#         if isinstance(dilation, int):
#             self.dilation = (dilation, dilation)
#         self.groups = groups
#         self.bias = bias
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         assert in_channels % groups == 0

#         self.kernel_len = self.kernel_size[0]*self.kernel_size[1]
#         self.center_idx = (self.kernel_len-1)//2
#         self.group_out_channels = self.kernel_len * in_channels // groups
#         self.new_shape = None

#         self.conv_query = nn.Conv2d(in_channels, in_channels*self.kernel_len, kernel_size, padding='same', groups=in_channels, bias=bias)
#         self.conv_key = nn.Conv2d(in_channels, in_channels*self.kernel_len, kernel_size, padding='same', groups=in_channels, bias=bias)
#         self.conv_value = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=bias)
#         self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        
#         self.softmax = nn.Softmax(-1)
#         self.reset_parameters()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         KH, KW = self.kernel_size
#         if self.new_shape is None:
#             self.new_shape = self._new_shape(H,W)
#             self.new_pixel = self.new_shape[0]*self.new_shape[1]
#         query = self.conv_query(x)
#         key = self.conv_key(x)
#         value = self.conv_value(x)
#         query = self.unfold(query)
#         key = self.unfold(key)
#         value = self.unfold(value)

#         query = query.reshape(B*self.groups, self.group_out_channels, self.kernel_len, self.new_pixel).transpose(1,3)
#         query = query[:,:,self.center_idx:self.center_idx+1]
#         key = key.reshape(B*self.groups, self.group_out_channels, self.kernel_len, self.new_pixel).transpose(1,3)
#         value = value.reshape(B*self.groups, self.out_channels//self.groups, self.kernel_len, self.new_pixel).transpose(1,3)

#         out = F._scaled_dot_product_attention(query,key,value)[0]
#         out = out.transpose(1,3).reshape(B, self.out_channels, *self.new_shape)
#         return out

#     def reset_parameters(self):
#         nn.init.kaiming_normal_(self.conv_query.weight, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.conv_key.weight, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.conv_value.weight, mode='fan_out', nonlinearity='relu')
    
#     def _new_shape(self, H, W):
#         SH,SW = self.stride
#         PH, PW = self.padding
#         DH, DW = self.dilation
#         KH, KW = self.kernel_size
#         H_new = (H + 2*PH - DH*(KH-1)-1)//SH + 1
#         W_new = (W + 2*PW - DW*(KW-1)-1)//SW + 1
#         return H_new, W_new


class NeighborAttention(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=4, bias=False, device=None):
        super(NeighborAttention, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, device)
        self.center_idx = (self.kernel_len-1)//2

        self.conv = nn.Conv2d(in_channels, 3*out_channels, kernel_size=1, padding=padding, bias=bias)
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        # self.rpe_h = nn.Parameter(torch.empty(groups, 1, self.kernel_size[0], 1, self.group_out_channels // 2))
        # self.rpe_w = nn.Parameter(torch.empty(groups, 1, 1, self.kernel_size[1], self.group_out_channels // 2))
        self.softmax = nn.Softmax(-1)
        # self.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        # KH, KW = self.kernel_size
        if self.new_shape is None:
            self.new_shape = self._new_shape(H,W)
            self.new_pixel = self.new_shape[0]*self.new_shape[1]
        
        x = self.conv(x)
        x = self.unfold(x)
        x = x.reshape(B, 3*self.groups, self.group_out_channels, self.kernel_len, self.new_pixel).transpose(2,4)
        query, key, value = x.split(self.groups, 1)

        query = query[:,:,:,self.center_idx:self.center_idx+1]

        # key = key.reshape(B, self.groups, self.new_pixel, KH, KW, self.group_out_channels)
        # key_h, key_w = key.split(self.group_out_channels // 2, dim=-1)
        # key = torch.concat((key_h + self.rpe_h, key_w + self.rpe_w), dim=1)
        # key = key.reshape(B, self.groups, self.new_pixel, self.kernel_len, self.group_out_channels)
        key = key.transpose(3,4)

        out = torch.matmul(query, key)
        out = self.softmax(out)
        out = torch.matmul(out, value)
        out = out.transpose(2,4).reshape(B, self.out_channels, *self.new_shape)
        return out

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.rpe_h, 0, 1)
        nn.init.normal_(self.rpe_w, 0, 1)


class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        query = self.query_conv(x)
        key = self.key_conv(padded_x)
        value = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        key = key.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        value = value.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        key = key[:, :, :H, :W, :, :]
        value = value[:, :, :, :H, :W, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        value = emb * value

        key = key.contiguous().view(B, self.groups, self.out_channels // self.groups, H, W, -1)
        value = value.contiguous().view(self.m, B, self.groups, self.out_channels // self.groups, H, W, -1)
        value = torch.sum(value, dim=0).view(B, self.groups, self.out_channels // self.groups, H, W, -1)

        query = query.view(B, self.groups, self.out_channels // self.groups, H, W, 1)

        out = query * key
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, value).view(B, -1, H, W)

        return out

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            nn.init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        nn.init.normal_(self.emb_a, 0, 1)
        nn.init.normal_(self.emb_b, 0, 1)
        nn.init.normal_(self.emb_mix, 0, 1)


class ABC_2D_Agnostic(nn.Module):
    def __init__(self, in_channel, kernel_number_per_pixel, kernel_size, hash, bias=False, batch_size=128, device=None):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.kernel_number_per_pixel = kernel_number_per_pixel
        self.batch_size = batch_size
        self.hash = self._build_full_hash(hash)
        self.if_bias = bias
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.weights = nn.Parameter(torch.empty(kernel_number_per_pixel, in_channel*kernel_size))
        self.bias = nn.Parameter(torch.empty(kernel_number_per_pixel, 1))
        nn.init.uniform_(self.weights, a=-np.sqrt(1/in_channel/kernel_size), b=np.sqrt(1/in_channel/kernel_size))
        nn.init.uniform_(self.bias, a=-np.sqrt(1/in_channel/kernel_size), b=np.sqrt(1/in_channel/kernel_size))

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.img_reconstruction(x)
        x = torch.matmul(self.weights, x)
        if self.if_bias:
            x = x + self.bias
        # knpp, B*H*W
        x = x.reshape(self.kernel_number_per_pixel,B,H,W).transpose(0,1)
        return x

    def img_reconstruction(self, x):
        B,C,H,W = x.shape
        if B > self.hash.shape[0]:
            raise ValueError('The batch size of input must be smaller than the defined batch_size or default value')
        hash = self.hash[:B]
        x = x.take(hash)
        # B, C, H, W, kernel_size
        x = x.permute(1,4,0,2,3).reshape(self.in_channel*self.kernel_size, B*H*W)
        return x

    def _build_full_hash(self, hashtable):
        HC, HH, HW, HHW = hashtable.shape
        if self.in_channel % HC !=0:
            raise ValueError('The defined in_channel has to be divisible by the first dimension of hashtable')
        if HH * HW !=HHW:
            raise ValueError('The last dimension of hash must be same as the second dimension times the third dimension')
        if HHW < self.kernel_size:
            raise ValueError('The defined kernel_size must smaller than hash-implied number of pixels')
        
        hashtable = hashtable.argsort(dim=-1, descending=True)
        hash = torch.empty((0))
        for channel in range(HC):
            channel_hash = hashtable[channel, :, :, :self.kernel_size]
            hash = torch.concat([hash, channel_hash.unsqueeze(0) + channel*HHW], axis=0)
        batch_hash = torch.empty((0))
        for r in range(int(self.in_channel/HC)):
            batch_hash = torch.concat([batch_hash, hash + r*HC*HHW], axis=0)
        full_hash = torch.empty((0))
        for bacth in range(self.batch_size):
            full_hash = torch.concat([full_hash, batch_hash.unsqueeze(0) + bacth*self.in_channel*HHW], axis=0)
        return full_hash.long().to(self.device)


class ABC_2D_Specific(nn.Module):
    def __init__(self, in_channel, kernel_number_per_pixel, kernel_size, hash, bias=False, batch_size=128, device=None):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.kernel_number_per_pixel = kernel_number_per_pixel
        self.batch_size = batch_size
        self.hash = self._build_full_hash(hash)
        self.if_bias = bias
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.weights = nn.Parameter(torch.empty(hash.shape[-1], kernel_number_per_pixel, in_channel*kernel_size))
        self.bias = nn.Parameter(torch.empty(1, kernel_number_per_pixel, 1))
        nn.init.uniform_(self.weights, a=-np.sqrt(1/in_channel/kernel_size), b=np.sqrt(1/in_channel/kernel_size))
        nn.init.uniform_(self.bias, a=-np.sqrt(1/in_channel/kernel_size), b=np.sqrt(1/in_channel/kernel_size))
        
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.img_reconstruction(x)
        x = torch.matmul(self.weights, x) 
        if self.if_bias:
            x = x + self.bias
        # H*W, knpp, B
        x = x.transpose(0,2).reshape(B, self.kernel_number_per_pixel, H, W)
        return x

    def img_reconstruction(self, x):
        B,C,H,W = x.shape
        if B > self.hash.shape[0]:
            raise ValueError('The batch size of input must be smaller than the defined batch_size or default value')
        hash = self.hash[:B]
        x = x.take(hash)
        # B, C, H, W, kernel_size
        x = x.permute(2,3,1,4,0).reshape(H*W, self.in_channel*self.kernel_size, B)
        return x
    
    def _build_full_hash(self, hashtable):
        HC, HH, HW, HHW = hashtable.shape
        if self.in_channel % HC !=0:
            raise ValueError('The defined in_channel has to be divisible by the first dimension of hashtable')
        if HH * HW !=HHW:
            raise ValueError('The last dimension of hash must be same as the second dimension times the third dimension')
        if HHW < self.kernel_size:
            raise ValueError('The defined kernel_size must smaller than hash-implied number of pixels')
        
        hashtable = hashtable.argsort(dim=-1, descending=True)
        hash = torch.empty((0))
        for channel in range(HC):
            channel_hash = hashtable[channel, :, :, :self.kernel_size]
            hash = torch.concat([hash, channel_hash.unsqueeze(0) + channel*HHW], axis=0)
        batch_hash = torch.empty((0))
        for r in range(int(self.in_channel/HC)):
            batch_hash = torch.concat([batch_hash, hash + r*HC*HHW], axis=0)
        full_hash = torch.empty((0))
        for bacth in range(self.batch_size):
            full_hash = torch.concat([full_hash, batch_hash.unsqueeze(0) + bacth*self.in_channel*HHW], axis=0)
        return full_hash.long().to(self.device)


class ABC_2D_Large(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, perceptual_size, hash, stride=(1,1), bias=False, batch_size=128, device=None):
        super().__init__()
        self.hash = hash
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.kernel_length = self.kernel_size[0]*self.kernel_size[1]
        self.perceptual_size = perceptual_size
        self.out_channel = out_channel
        self.if_bias = bias
        self.batch_size = batch_size
        self.stride = stride
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv_hash, self.zerofy_hash = self._build_hash(hash)
        self.weights = nn.Parameter(torch.empty(out_channel, in_channel*self.kernel_length))
        self.bias = nn.Parameter(torch.empty(out_channel, 1))
        nn.init.uniform_(self.weights, a=-np.sqrt(1/in_channel/self.perceptual_size), b=np.sqrt(1/in_channel/self.perceptual_size))
        nn.init.uniform_(self.bias, a=-np.sqrt(1/in_channel/self.perceptual_size), b=np.sqrt(1/in_channel/self.perceptual_size))
    
    def forward(self, x):
        B,C,H,W = x.shape
        _, C, NH, NW, ks = self.conv_hash.shape
        x = self.img_reconstruction(x)
        x = torch.matmul(self.weights, x)
        if self.if_bias:
            x = x + self.bias
        # out_channel, B*NH*NW
        x = x.reshape(self.out_channel,B,NH,NW).transpose(0,1)
        return x

    def img_reconstruction(self, x):
        B,C,H,W = x.shape
        if B > self.conv_hash.shape[0]:
            raise ValueError('The batch size of input must be smaller than the defined batch_size or default value')
        conv_hash = self.conv_hash[:B]
        zerofy_hash = self.zerofy_hash[:B]
        B, C, NH, NW, kl = conv_hash.shape
        x = x.take(conv_hash)
        x[zerofy_hash==1] = 0
        # B, C, H, W, kernel_size
        x = x.permute(1,4,0,2,3).reshape(self.in_channel*self.kernel_length, B*NH*NW)
        return x
    
    def _build_hash(self, hashtable):
        hashtable = hashtable.argsort(dim=-1, descending=True)
        HC, HH, HW, HHW = hashtable.shape
        KH, KW = self.kernel_size
        if self.in_channel % HC !=0:
            raise ValueError('The defined in_channel has to be divisible by the first dimension of hashtable')
        if HH * HW !=HHW:
            raise ValueError('The last dimension of hash must be same as the second dimension times the third dimension')
        if (HH < self.kernel_size[0]) or (HW < self.kernel_size[1]):
            raise ValueError('The defined kernel_size must smaller than hash-implied image size')

        self.new_hash = hashtable.unflatten(-1,(HH,HW))[:,::self.stride[0],::self.stride[1],::self.stride[0],::self.stride[1]].flatten(-2,-1)

        HH_new = int(HH/self.stride[0])
        HW_new = int(HW/self.stride[1])
        batch_conv_hash_t = torch.empty((0))
        batch_zerofy_hash_t = torch.empty((0))
        for c in range(HC):
            channel_conv_hash = torch.empty((0))
            channel_zerofy_hash = torch.empty((0))
            for h in range(int(HH/self.stride[0])):
                h = h * self.stride[0]
                for w in range(int(HW/self.stride[1])):
                    w = w * self.stride[1]
                    n = 0
                    pixel_conv_hash = torch.zeros((KH, KW))
                    pixel_zerofy_hash = torch.ones((KH, KW))
                    for i in hashtable[c, h, w, :]:
                        if n < self.perceptual_size:
                            irh = torch.div(i, HH, rounding_mode='floor') - h
                            irw = i%HH - w
                            if (abs(irh) <= (KH-1)/2) and (abs(irw) <= (KW-1)/2):
                                pixel_conv_hash[int(irh + (KH-1)/2), int(irw + (KW-1)/2)] = i
                                pixel_zerofy_hash[int(irh + (KH-1)/2), int(irw + (KW-1)/2)] = 0
                                n = n + 1
                    channel_conv_hash = torch.concat([channel_conv_hash, pixel_conv_hash.reshape(1, KH*KW)], axis=0)
                    channel_zerofy_hash = torch.concat([channel_zerofy_hash, pixel_zerofy_hash.reshape(1, KH*KW)], axis=0)
            batch_conv_hash_t = torch.concat([batch_conv_hash_t, channel_conv_hash.reshape(1, HH_new, HW_new, KH*KW) + c * HH * HW])
            batch_zerofy_hash_t = torch.concat([batch_zerofy_hash_t, channel_zerofy_hash.reshape(1, HH_new, HW_new, KH*KW)])
        
        batch_conv_hash = torch.empty((0))
        batch_zerofy_hash = torch.empty((0))
        for r in range(int(self.in_channel/HC)):
            batch_conv_hash = torch.concat([batch_conv_hash, batch_conv_hash_t + r*HC*HH*HW], axis=0)
            batch_zerofy_hash = torch.concat([batch_zerofy_hash, batch_zerofy_hash_t], axis=0)

        conv_hash = torch.empty((0))
        zerofy_hash = torch.empty((0))
        for b in range(self.batch_size):
            conv_hash = torch.concat([conv_hash, batch_conv_hash.unsqueeze(0)+b*self.in_channel*HH*HW])
            zerofy_hash = torch.concat([zerofy_hash, batch_zerofy_hash.unsqueeze(0)])
        return conv_hash.long().to(self.device), zerofy_hash.to(self.device)