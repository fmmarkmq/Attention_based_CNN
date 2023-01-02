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
from data.data_loader import ABC_Data_Loader

class ABC_Driver(object):
    def __init__(self, args, data=None):
        self.args = args
        self.data = data
        self.device = self._acquire_device()
        self.data_loader = self._build_data_loader()
        self.model = self._build_model()
        

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
            for idx, (inputs, labels) in enumerate(train_loader):
                inputs=inputs.to(device)
                labels=labels.to(device)
                model_optim.zero_grad(set_to_none = True)
                preds = model(inputs)
                loss = criterion(preds,labels)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            print(f'epoch: {epoch}, train_loss: {train_loss}')
        return self

    def predict(self, pred_loader=None):
        if pred_loader is None:
            pred_loader = self.data_loader.predict 
        model =self.model
        device = self.device
        
        model.eval()
        preds = torch.tensor([])
        for idx, (inputs, labels) in enumerate(pred_loader):
            pred = model(inputs.to(device)).cpu().detach()
            preds = torch.concat([preds, pred])
        return preds

    def metric(self, pred_loader=None):
        if pred_loader is None:
            pred_loader = self.data_loader.predict 
        y_true = pred_loader.dataset.targets
        y_pred = self.predict(pred_loader)
        return accuracy_score(y_true, y_pred.argmax(axis=1))
    
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
        data_loader = ABC_Data_Loader(self.args, self.data)
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

    def get_cov_hashTable(self, data_mat:torch.tensor):
            data_shape = data_mat.shape
            data_mat=  data_mat.reshape(data_shape[0], -1, data_shape[-2], data_shape[-1])
            B,C,H,W = data_mat.shape
            idx_list_channels = []
            for channel in range(C):
                data_mat1 = data_mat[:,channel,:]
                Num, Hi, Wi = data_mat1.shape
                data_mat1 = data_mat1.reshape(-1, Hi*Wi).T
                cov = torch.cov(data_mat1).abs()
                var = torch.var(data_mat1.to(torch.float32), axis=1).abs()
                var[var<0.01] = var[var<0.01] + 1
                corr = (cov/var.pow(0.5)).T/var.pow(0.5)
                val,idx = torch.topk(corr,k=self.args.kernel_size,dim=0,sorted=True,largest=True)
                idx_list_channels.append(idx.T+channel*H*W)
            full_idx_list = torch.concat(idx_list_channels, axis=1)
            full_idx = torch.empty((0))
            for i in range(self.args.input_channel):
                full_idx = torch.concat([full_idx, full_idx_list + i*H*W], axis=1)
            all_idx = torch.empty((0))
            for batch in range(self.args.train.batch_size):
                all_idx = torch.concat([all_idx, full_idx + batch*C*H*W], axis=0)
            return all_idx