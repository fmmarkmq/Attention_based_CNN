from operator import concat
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from statsmodels.tsa.api import VAR
import gc
import numpy as np


class ABC_2D(nn.Module):
    def __init__(self, in_channel, kernel_size, pixel_number, kernel_number_per_pixel, batch_size=100, hash=None):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.pixel_number = pixel_number
        self.kernel_number_per_pixel = kernel_number_per_pixel
        self.batch_size = batch_size
        self.hash = self._build_full_hash(hash)
        # self.weights = nn.Parameter(torch.empty(pixel_number, kernel_number_per_pixel, in_channel*kernel_size))
        self.weights = nn.Parameter(torch.empty(kernel_number_per_pixel, in_channel*kernel_size))
        # nn.init.normal_(self.weights)
        nn.init.uniform_(self.weights, a=-np.sqrt(1/in_channel/kernel_size), b=np.sqrt(1/in_channel/kernel_size))
        
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.img_reconstruction(self.hash, x)
        w_times_x = self.weights.matmul(x)
        # w_times_x = w_times_x.transpose(0,2).reshape(B, -1, H, W)
        w_times_x = w_times_x.reshape(self.kernel_number_per_pixel,B,H,W).transpose(0,1)
        return w_times_x

    def img_reconstruction(self, hashtable, img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B,C,H,W = img.shape
        if self.pixel_number*B<= len(hashtable):
            hash = hashtable[:self.pixel_number*B]
        else:
            hashtable = hashtable[:self.pixel_number]
            hash = torch.empty((0))
            for batch in range(B):
                hash = torch.concat([hash, hashtable + batch*C*H*W])
        # final_img = img.take(hash.reshape(-1).long().to(device)).reshape(B, H*W, -1).permute(1,2,0)
        final_img = img.take(hash.reshape(-1).long().to(device)).reshape(B, H*W, -1).permute(2,0,1).reshape(-1, B*H*W)
        return final_img
    
    def _build_full_hash(self, hashtable):
        HH,HW = hashtable.shape
        if self.kernel_size*self.in_channel <= HW:
            hash = hashtable[:, :self.kernel_size*self.in_channel]
        else:
            hash = torch.empty((0))
            for channel in range(int(self.kernel_size*self.in_channel/HW)):
                hash = torch.concat([hash, hashtable + channel*HW/self.kernel_size*self.pixel_number], axis=1)
        full_hash = torch.empty((0))
        for bacth in range(self.batch_size):
            full_hash = torch.concat([full_hash, hash + bacth*self.in_channel*self.pixel_number], axis=0)
        return full_hash


# class ABC_2D_Large(nn.Module):
#     def __init__(self, in_channel, kernel_size, pixel_number, kernel_number_per_pixel, hash=None):
#         super().__init__()
#         self.hash = hash
#         self.in_channel = in_channel
#         self.kernel_size = kernel_size
#         self.pixel_number = pixel_number
#         self.kernel_number_per_pixel = kernel_number_per_pixel
#         self.weights = nn.Parameter(torch.empty(kernel_number_per_pixel, in_channel*kernel_size))
#         nn.init.normal_(self.weights)
#         self.register_parameter('weights', self.weights)