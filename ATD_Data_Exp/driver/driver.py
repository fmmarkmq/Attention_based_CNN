import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from model.model import ABC_Net
from data.data_loader import ABC_data_loader

class ABC_Driver(object):
    def __init__(self, args, df:pd.DataFrame):
        self.data = df
        self.args = args
        self.device = self._acquire_device()
        self.data_loader = self._build_data_loader()
        self.model = self._build_model().to(self.device)
        # self.model = self._build_model()
        

    def train(self, train_loader=None):
        if train_loader is None:
            train_loader = self.data_loader.train
        model =self.model
        device = self.device
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        model.train()
        for epoch in range(self.args.train_epochs):
            train_loss=[]
            # start_time = time.time()
            for idx, (inputs, labels) in enumerate(train_loader):
                inputs=inputs.to(torch.float32)
                labels=labels.to(torch.float32)
                inputs=inputs.to(device)
                labels=labels.to(device)
                model_optim.zero_grad(set_to_none = True)
                # print("label:", labels.shape)
                # print(inputs.shape)
                preds = model(inputs)
                loss = criterion(preds,labels)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            print(f'epoch: {epoch}, train_loss: {train_loss}')
        return self

    # def predict(self, pred_loader=None):
    #     if pred_loader is None:
    #         pred_loader = self.data_loader.predict 
    #     model =self.model
    #     device = self.device
        
    #     model.eval()
    #     preds = torch.tensor([])
    #     for idx, (inputs, labels) in enumerate(pred_loader):
    #         pred = model(inputs.to(device)).cpu().detach()
    #         preds = torch.concat([preds, pred])
    #     return preds

    # def update_df(self, new_rows):
    #     #tmp = self.df.drop(["timeStamps"], axis=1)
    #     tmp = self.df
    #     last_idx=tmp.index[-len(new_rows):]
    #     new_df = pd.DataFrame(new_rows, index=last_idx+self.args.predict_len, columns=tmp.columns)
    #     tmp = pd.concat([tmp, new_df])

        
    #     #tmp.insert(0, "timeStamps", tmp.index)
    #     self.df = tmp
    #     return

    def predict(self, pred_loader=None):
        if pred_loader is None:
            pred_loader = self.data_loader.predict
        model =self.model
        device = self.device

        model.eval()
        preds = torch.tensor([])
        for idx, (inputs)in enumerate(pred_loader):
            inputs=inputs.to(torch.float32)
            pred = model(inputs.to(device)).cpu().detach()
            preds=torch.concat([preds, pred])
        return preds



    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        
        self.device = device
        return device
    
    def _build_model(self):
        self.hash = self.get_cov_hashTable(self.data_loader.train.dataset.data)
        model = ABC_Net(self.args, self.hash).to(self.device)
        return model
    
    def _build_data_loader(self):
        data_loader = ABC_data_loader(self.args, self.data)
        return data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        # model_optim = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        return model_optim
    
    def _select_criterion(self):
        if self.args.criterion == 'L1':
            criterion =  nn.L1Loss()
        elif self.args.criterion == 'CE':
            criterion = nn.CrossEntropyLoss()
        elif self.args.criterion == 'nll':
            criterion = nn.NLLLoss()
        elif self.args.criterion == "mse":
            criterion = nn.MSELoss()
        return criterion


    def get_surrounding_pixel_indices(self, grid, center_index):
        # Calculate the number of rows and columns in the grid
        num_rows = len(grid)
        num_cols = len(grid[0])
        
        # Calculate the row and column indices of the center pixel
        center_row = center_index // num_cols
        center_col = center_index % num_cols
        
        # Initialize an empty list to store the indices of the surrounding pixels
        surrounding_pixel_indices = []
        
        # Iterate over the rows and columns in a 3x3 grid centered on the center pixel
        for row in range(center_row-1, center_row+2):
            for col in range(center_col-1, center_col+2):
                # Check if the current row and column indices are valid (i.e., within the bounds of the grid)
                if (row >= 0 and row < num_rows) and (col >= 0 and col < num_cols):
                    # If the current row and column indices are valid, calculate the index of the pixel at that position and add it to the list of surrounding pixel indices
                    surrounding_pixel_indices.append(row * num_cols + col)
        
        # Return the list of surrounding pixel indices, excluding the index of the center pixel itself
        return [index for index in surrounding_pixel_indices if index != center_index]

    def xy_to_idx(self, x: int, y: int, board_size: int = 28) -> int:
        return x * board_size + y

    # def get_cov_hashTable(self, data_mat):
    #     data_shape = data_mat.shape
    #     data_mat=  data_mat.reshape(data_shape[0], -1, data_shape[-2], data_shape[-1])
    #     B,C,H,W = data_mat.shape
    #     idx_list_channels = []
    #     for channel in range(C):
    #         for i in range(2,27):
    #             for j in range(2,27):
    #                 center_idx = self.xy_to_idx(i,j)
    #                 idx_lst = self.get_surrounding_pixel_indices(np.empty([H,W]), center_idx)
    #                 idx_lst.append(center_idx)
    # #                 print(center_idx, idx_lst)
    #                 idx_list_channels.append(torch.tensor([idx_lst]))
    #     return {int(row[0][-1]): row for i, row in enumerate(idx_list_channels)}

    def get_cov_hashTable(self, data_mat:torch.tensor):
        data_shape = data_mat.shape
        # print(data_shape)
        data_mat=  data_mat.reshape(data_shape[0], -1, data_shape[-2], data_shape[-1])
        B,C,H,W = data_mat.shape
        idx_list_channels = []
        for channel in range(C):
            data_mat1 = data_mat[:,channel,:]
            Num, Hi, Wi = data_mat1.shape
            data_mat1 = data_mat1.reshape(-1, Hi*Wi)
            cov  = torch.cov(data_mat1.T).abs()
            val,idx = torch.topk(cov,k=self.args.kernel_size,dim=0,sorted=True,largest=True)
            # idx_expanded = torch.unsqueeze(idx.T, axis = 1)
            # print(idx_expanded.shape)
            idx_list_channels.append(idx.T+channel*H*W)
        full_idx_list = torch.concat(idx_list_channels, axis=1)
        full_idx = torch.empty((0))
        for i in range(self.args.input_channel):
            full_idx = torch.concat([full_idx, full_idx_list + i*H*W], axis=1)
        # return {i: row for i, row in enumerate(full_idx_list)}
        all_idx = torch.empty((0))
        for batch in range(self.args.batch_size):
            all_idx = torch.concat([all_idx, full_idx + batch*C*H*W], axis=0)
        return all_idx

        # N, HI, WI = data_mat.shape
        # data_mat = data_mat.reshape(-1, HI*WI)
        # cov  = torch.cov(data_mat.T).abs()
        # val,idx = torch.topk(cov,k=9,dim=0,sorted=True,largest=True)
        # hashtable = {i: row for i, row in enumerate(idx.T)}
        # return hashtable