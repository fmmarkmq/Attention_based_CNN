from operator import concat
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from statsmodels.tsa.api import VAR
import gc


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
        # if multi_channel == False:
        #     BA,HI,WI = img.shape
        #     sigle_img_idx = torch.empty((0)).to(device)
        #     for key in hashtable.keys():
        #         sigle_img_idx = torch.concat([sigle_img_idx, hashtable[key]])
        #     all_idx = torch.empty((0)).to(device)
        #     for batch in range(BA):
        #         all_idx = torch.concat([all_idx, sigle_img_idx + HI*WI*batch])
        #     return img.take(all_idx.long()).reshape(-1, HI*WI, B)

        B,C,H,W = img.shape
        # H=25
        # W=25
        sigle_img_idx = torch.empty((0))
        for key in hashtable.keys():
            idx = hashtable[key]
            for channel in range(idx.shape[0]):
                sigle_img_idx = torch.concat([sigle_img_idx, idx[channel] + H*W*channel])
        all_idx = torch.empty((0))
        for batch in range(B):
            all_idx = torch.concat([all_idx, sigle_img_idx + H*W*C*batch])
        # final_img = img.take(all_idx.long().to(device)).reshape(-1, H*W, B).transpose(0,1)
        # final_img = img.take(all_idx.long().to(device)).reshape(-1, H*W, B).transpose(0,1)
        final_img = img.take(all_idx.long().to(device)).reshape(B, H*W, -1).permute(1,2,0)
        return final_img
