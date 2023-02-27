import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math


# class AttentionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
#         super(AttentionConv, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups

#         assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

#         self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
#         self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

#         self.reset_parameters()

#     def forward(self, x):
#         batch, channels, height, width = x.size()

#         padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
#         q_out = self.query_conv(x)
#         k_out = self.key_conv(padded_x)
#         v_out = self.value_conv(padded_x)

#         k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

#         k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
#         k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

#         k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

#         q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

#         out = q_out * k_out
#         out = F.softmax(out, dim=-1)
#         out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

#         return out

#     def reset_parameters(self):
#         init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

#         init.normal_(self.rel_h, 0, 1)
#         init.normal_(self.rel_w, 0, 1)


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=4, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        if isinstance(stride, int):
            self.stride = (stride, stride)
        self.padding = padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        self.dilation = dilation
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        self.groups = groups
        self.bias = bias
        assert out_channels % groups == 0
        self.group_out_channels = out_channels // groups

        self.kernel_len = self.kernel_size[0]*self.kernel_size[1]
        self.center_idx = (self.kernel_len-1)//2
        self.new_shape = None

        self.conv = nn.Conv2d(in_channels, 3*out_channels, kernel_size=1, padding=padding, bias=bias)
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.rpe_h = nn.Parameter(torch.empty(groups, 1, self.kernel_size[0], 1, self.group_out_channels // 2))
        self.rpe_w = nn.Parameter(torch.empty(groups, 1, 1, self.kernel_size[1], self.group_out_channels // 2))
        self.softmax = nn.Softmax(-1)
        self.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        KH, KW = self.kernel_size
        if self.new_shape is None:
            self.new_shape = self._new_shape(H,W)
            self.new_pixel = self.new_shape[0]*self.new_shape[1]
        
        x = self.conv(x)
        x = self.unfold(x)
        x = x.reshape(B, 3*self.groups, self.group_out_channels, self.kernel_len, self.new_pixel).transpose(2,4)
        query, key, value = x.split(self.groups, 1)

        query = query[:,:,:,self.center_idx:self.center_idx+1]

        key = key.reshape(B, self.groups, self.new_pixel, KH, KW, self.group_out_channels)
        key_h, key_w = key.split(self.group_out_channels // 2, dim=-1)
        key = torch.concat((key_h + self.rpe_h, key_w + self.rpe_w), dim=1)
        key = key.reshape(B, self.groups, self.new_pixel, self.kernel_len, self.group_out_channels)
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
    
    def _new_shape(self, H, W):
        SH,SW = self.stride
        PH, PW = self.padding
        DH, DW = self.dilation
        KH, KW = self.kernel_size
        H_new = (H + 2*PH - DH*(KH-1)-1)//SH + 1
        W_new = (W + 2*PW - DW*(KW-1)-1)//SW + 1
        return H_new, W_new


class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


# temp = torch.randn((2, 3, 32, 32))
# conv = AttentionConv(3, 16, kernel_size=3, padding=1)
# print(conv(temp).size())
