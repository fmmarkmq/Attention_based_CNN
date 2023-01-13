from operator import concat
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from statsmodels.tsa.api import VAR
import numpy as np
import gc


class ABC_2D_Agnotic(nn.Module):
    def __init__(self, in_channel, pixel_number, kernel_size, kernel_number_per_pixel, hash, batch_size=100, if_bias=False):
        super().__init__()
        self.in_channel = in_channel
        self.pixel_number = pixel_number
        self.kernel_size = kernel_size
        self.kernel_number_per_pixel = kernel_number_per_pixel
        self.batch_size = batch_size
        self.hash = self._build_full_hash(hash)
        self.if_bias = if_bias
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        HC, HH, HW, HHW = hashtable.shape
        if self.in_channel % HC !=0:
            raise ValueError('The defined in_channel has to be divisible by the first dimension of hashtable')
        if HH * HW !=HHW:
            raise ValueError('The last dimension of hash must be same as the second dimension times the third dimension')
        if HHW != self.pixel_number:
            raise ValueError('The defined pixel_number and hash-implied number of pixels are not consistent')
        if HHW < self.kernel_size:
            raise ValueError('The defined kernel_size must smaller than hash-implied number of pixels')
        
        hash = torch.empty((0))
        for channel in range(HC):
            channel_hash = hashtable[channel, :, :, :self.kernel_size]
            hash = torch.concat([hash, channel_hash.unsqueeze(0) + channel*self.pixel_number], axis=0)
        batch_hash = torch.empty((0))
        for r in range(int(self.in_channel/HC)):
            batch_hash = torch.concat([batch_hash, hash + r*HC*self.pixel_number], axis=0)
        full_hash = torch.empty((0))
        for bacth in range(self.batch_size):
            full_hash = torch.concat([full_hash, batch_hash.unsqueeze(0) + bacth*self.in_channel*self.pixel_number], axis=0)
        return full_hash.long().to(device)



class ABC_2D_Specific(nn.Module):
    def __init__(self, in_channel, pixel_number, kernel_size, kernel_number_per_pixel, hash, batch_size=100, if_bias=False):
        super().__init__()
        self.in_channel = in_channel
        self.pixel_number = pixel_number
        self.kernel_size = kernel_size
        self.kernel_number_per_pixel = kernel_number_per_pixel
        self.batch_size = batch_size
        self.hash = self._build_full_hash(hash)
        self.if_bias = if_bias
        self.weights = nn.Parameter(torch.empty(pixel_number, kernel_number_per_pixel, in_channel*kernel_size))
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
        x = x.permute(2,3,1,4,0).reshape(self.pixel_number, self.in_channel*self.kernel_size, B)
        return x
    
    def _build_full_hash(self, hashtable):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        HC, HH, HW, HHW = hashtable.shape
        if self.in_channel % HC !=0:
            raise ValueError('The defined in_channel has to be divisible by the first dimension of hashtable')
        if HH * HW !=HHW:
            raise ValueError('The last dimension of hash must be same as the second dimension times the third dimension')
        if HHW != self.pixel_number:
            raise ValueError('The defined pixel_number and hash-implied number of pixels are not consistent')
        if HHW < self.kernel_size:
            raise ValueError('The defined kernel_size must smaller than hash-implied number of pixels')
        
        hash = torch.empty((0))
        for channel in range(HC):
            channel_hash = hashtable[channel, :, :, :self.kernel_size]
            hash = torch.concat([hash, channel_hash.unsqueeze(0) + channel*self.pixel_number], axis=0)
        batch_hash = torch.empty((0))
        for r in range(int(self.in_channel/HC)):
            batch_hash = torch.concat([batch_hash, hash + r*HC*self.pixel_number], axis=0)
        full_hash = torch.empty((0))
        for bacth in range(self.batch_size):
            full_hash = torch.concat([full_hash, batch_hash.unsqueeze(0) + bacth*self.in_channel*self.pixel_number], axis=0)
        return full_hash.long().to(device)



class ABC_2D_Large(nn.Module):
    def __init__(self, in_channel, kernel_size, perceptual_size, out_channel, hash, batch_size=100):
        super().__init__()
        self.hash = hash
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.kernel_length = self.kernel_size[0]*self.kernel_size[1]
        self.perceptual_size = perceptual_size
        self.out_channel = out_channel
        self.batch_size = batch_size
        self.conv_hash, self.zerofy_hash = self._build_hash(hash)
        self.weights = nn.Parameter(torch.empty(out_channel, in_channel*self.kernel_length))
        self.bias = nn.Parameter(torch.empty(out_channel, 1))
        nn.init.uniform_(self.weights, a=-np.sqrt(1/in_channel/self.perceptual_size), b=np.sqrt(1/in_channel/self.perceptual_size))
        nn.init.uniform_(self.bias, a=-np.sqrt(1/in_channel/self.perceptual_size), b=np.sqrt(1/in_channel/self.perceptual_size))
    
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.img_reconstruction(x)
        x = torch.matmul(self.weights, x)
        # out_channel, B*H*W
        x = x.reshape(self.out_channel,B,H,W).transpose(0,1)
        return x

    def img_reconstruction(self, x):
        B,C,H,W = x.shape
        if B > self.conv_hash.shape[0]:
            raise ValueError('The batch size of input must be smaller than the defined batch_size or default value')
        conv_hash = self.conv_hash[:B]
        zerofy_hash = self.zerofy_hash[:B]
        x = x.take(conv_hash)
        x[zerofy_hash==1] = 0
        # B, C, H, W, kernel_size
        x = x.permute(1,4,0,2,3).reshape(self.in_channel*self.kernel_length, B*H*W)
        return x
    
    def _build_hash(self, hashtable):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        HC, HH, HW, HHW = hashtable.shape
        KH, KW = self.kernel_size
        if self.in_channel % HC !=0:
            raise ValueError('The defined in_channel has to be divisible by the first dimension of hashtable')
        if HH * HW !=HHW:
            raise ValueError('The last dimension of hash must be same as the second dimension times the third dimension')
        if (HH < self.kernel_size[0]) or (HW < self.kernel_size[1]):
            raise ValueError('The defined kernel_size must smaller than hash-implied image size')

        batch_conv_hash_t = torch.empty((0))
        batch_zerofy_hash_t = torch.empty((0))
        for c in range(HC):
            channel_conv_hash = torch.empty((0))
            channel_zerofy_hash = torch.empty((0))
            for h in range(HH):
                for w in range(HW):
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
            batch_conv_hash_t = torch.concat([batch_conv_hash_t, channel_conv_hash.reshape(1, HH, HW, KH*KW) + c * HH * HW])
            batch_zerofy_hash_t = torch.concat([batch_zerofy_hash_t, channel_zerofy_hash.reshape(1, HH, HW, KH*KW)])
        
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
        return conv_hash.long().to(device), zerofy_hash.to(device)



        # channel_cnn_hash = self.get_cnn_hashTable((HH, HW))
        # batch_cnn_hash = torch.empty((0))
        # for c in range(HC):
        #     batch_cnn_hash = torch.concat([batch_cnn_hash, channel_cnn_hash.unsqueeze(0) + c * HH * HW], axis=1)
        # cnn_hash = torch.empty((0))
        # for b in range(batch_size):
        #     cnn_hash = torch.concat([cnn_hash, batch_cnn_hash.unsqueeze(0) + b * HC * HH * HW])
        
    # def get_surrounding_pixel_indices(self, img_size: tuple, center_loc:tuple, kernel_size: tuple):
    #     num_rows, num_cols = img_size
    #     KH, KW = kernel_size
    #     center_row, center_col = center_loc 

    #     surrounding_pixel_indices = -np.ones(KH, KW)
    #     for row in range(KH):
    #         if (center_row-int((KH-1)/2) + row >= 0 and center_row-int((KH-1)/2) + row < num_rows):
    #             for col in range(KW):
    #                 if(center_col-int((KW-1)/2) + col >= 0 and center_col-int((KW-1)/2) + col < num_cols):
    #                     surrounding_pixel_indices[row, col] = (row * num_cols + col)
    #     return surrounding_pixel_indices.reshape(1, KH*KW)

    # def get_cnn_hashTable(self, img_size: tuple):
    #     H,W = img_size
    #     KH, KW = self.kernel_size
    #     hash = np.empty((0))
    #     for i in range(H):
    #         for j in range(W):
    #             idx_lst = self.get_surrounding_pixel_indices((H,W), (i,j), (KH, KW))
    #             hash = np.concat([hash, idx_lst], axis=0)
    #     return hash.reshape(H, W, KH*KH)



# class ABC_2D(nn.Module):
#     def __init__(self, in_channel, kernel_size, pixel_number, kernel_number_per_pixel, batch_size=100, hash=None):
#         super().__init__()
#         self.in_channel = in_channel
#         self.kernel_size = kernel_size
#         self.pixel_number = pixel_number
#         self.kernel_number_per_pixel = kernel_number_per_pixel
#         self.batch_size = batch_size
#         self.hash = self._build_full_hash(hash)
#         # self.weights = nn.Parameter(torch.empty(pixel_number, kernel_number_per_pixel, in_channel*kernel_size))
#         self.weights = nn.Parameter(torch.empty(kernel_number_per_pixel, in_channel*kernel_size))
#         # nn.init.normal_(self.weights)
#         nn.init.uniform_(self.weights, a=-np.sqrt(1/in_channel/kernel_size), b=np.sqrt(1/in_channel/kernel_size))
        
#     def forward(self, x):
#         B,C,H,W = x.shape
#         x = self.img_reconstruction(self.hash, x)
#         w_times_x = self.weights.matmul(x)
#         # w_times_x = w_times_x.transpose(0,2).reshape(B, -1, H, W)
#         w_times_x = w_times_x.reshape(self.kernel_number_per_pixel,B,H,W).transpose(0,1)
#         return w_times_x

#     def img_reconstruction(self, hashtable, img):
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         B,C,H,W = img.shape
#         if self.pixel_number*B<= len(hashtable):
#             hash = hashtable[:self.pixel_number*B]
#         else:
#             hashtable = hashtable[:self.pixel_number]
#             hash = torch.empty((0))
#             for batch in range(B):
#                 hash = torch.concat([hash, hashtable + batch*C*H*W])
#         # final_img = img.take(hash.reshape(-1).long().to(device)).reshape(B, H*W, -1).permute(1,2,0)
#         final_img = img.take(hash.reshape(-1).long().to(device)).reshape(B, H*W, -1).permute(2,0,1).reshape(-1, B*H*W)
#         return final_img
    
#     def _build_full_hash(self, hashtable):
#         hashtable = hashtable.reshape(784, 784)[:, :9]
#         HH,HW = hashtable.shape
#         if self.kernel_size*self.in_channel <= HW:
#             hash = hashtable[:, :self.kernel_size*self.in_channel]
#         else:
#             hash = torch.empty((0))
#             for channel in range(int(self.kernel_size*self.in_channel/HW)):
#                 hash = torch.concat([hash, hashtable + channel*HW/self.kernel_size*self.pixel_number], axis=1)
#         full_hash = torch.empty((0))
#         for bacth in range(self.batch_size):
#             full_hash = torch.concat([full_hash, hash + bacth*self.in_channel*self.pixel_number], axis=0)
#         return full_hash
