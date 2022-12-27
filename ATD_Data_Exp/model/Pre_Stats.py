import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cov_hashTable(data_mat:torch.tensor, multi_channel=False):
    
    if multi_channel == True:
        B,C,H,W = data_mat.shape
        idx_list_channels = []
        for channel in range(C):
            data_mat1 = data_mat[:,channel,:]
            Num, Hi, Wi = data_mat1.shape
            data_mat1 = data_mat1.reshape(-1, Hi*Wi)
            cov  = torch.cov(data_mat1.T).abs()
            val,idx = torch.topk(cov,k=9,dim=0,sorted=True,largest=True)
            idx_expanded = torch.unsqueeze(idx.T, axis = 1)
            idx_list_channels.append(idx_expanded)
        full_idx_list = torch.concat(idx_list_channels, axis=1)
        return {i: row for i, row in enumerate(full_idx_list)}

    N, HI, WI = data_mat.shape
    data_mat = data_mat.reshape(-1, HI*WI)
    cov  = torch.cov(data_mat.T).abs()
    val,idx = torch.topk(cov,k=9,dim=0,sorted=True,largest=True)
    hashtable = {i: row for i, row in enumerate(idx.T)}
    return hashtable

def img_reconstruction(hashtable, img, multi_channel=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if multi_channel == True:
        B,C,H,W = img.shape
        sigle_img_idx = torch.empty((0)).to(device)
        for key in hashtable.keys():
            for channel in range(C):
                sigle_img_idx = torch.concat([sigle_img_idx, hashtable[key][channel] + H*W*channel])
        all_idx = torch.empty((0)).to(device)
        for batch in range(B):
            all_idx = torch.concat([all_idx, sigle_img_idx + H*W*C*batch])
        return img.take(all_idx.long()).reshape(-1, H*W, B)

    BA,HI,WI = img.shape
    sigle_img_idx = torch.empty((0)).to(device)
    for key in hashtable.keys():
        sigle_img_idx = torch.concat([sigle_img_idx, hashtable[key]])
    all_idx = torch.empty((0)).to(device)
    for batch in range(BA):
        all_idx = torch.concat([all_idx, sigle_img_idx + H*W*batch])
    return img.take(all_idx.long()).reshape(-1, H*W, B)
