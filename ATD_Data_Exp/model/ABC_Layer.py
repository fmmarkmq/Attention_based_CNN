from operator import concat
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from statsmodels.tsa.api import VAR
import gc
import time

class ABC_2D(nn.Module):
    def __init__(self, in_channel, kernel_size, pixel_number, kernel_number_per_pixel, hash=None):
        super().__init__()
        self.hash = hash
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.pixel_number = pixel_number
        self.kernel_number_per_pixel = kernel_number_per_pixel
        self.weights = nn.Parameter(torch.empty(pixel_number, kernel_number_per_pixel, in_channel*kernel_size))
        nn.init.normal_(self.weights)
        # self.register_parameter('weights', self.weights)

        
    def forward(self, x):
        x = self.img_reconstruction(self.hash, x, multi_channel=True)
        w_times_x = torch.matmul(self.weights, x)
        return w_times_x.transpose(0,2)
    
    def img_reconstruction(self, hashtable, img, multi_channel=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B,C,H,W = img.shape
        hashtable = hashtable[:self.pixel_number*B].reshape(-1)
        final_img = img.take(hashtable.long().to(device)).reshape(B, H*W, -1).permute(1,2,0)
        return final_img
