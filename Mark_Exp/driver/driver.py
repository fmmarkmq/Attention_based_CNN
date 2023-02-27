import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
import torch.nn.functional as F
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from model.model import ABC_Net
from data.data_loader import ABC_Data_Loader
from record.record import EXPERecords
from model.Stand_Alone_Self_Attention.model import ResNet26

class ABC_Driver(object):
    def __init__(self, args, data=None, record_path=None, if_hash=False, if_gpu=True):
        self.args = args
        self.data = data
        self.record_path = record_path
        self.if_gpu = if_gpu
        self.if_hash = if_hash
        self.device = self._acquire_device()
        self.data_loader = self._build_data_loader()
        self.model = self._build_model()
        self.record = self._build_record()

    def train(self, train_loader=None):
        if train_loader is None:
            train_loader = self.data_loader.train 
        model =self.model
        device = self.device
        criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
            

        for epoch in range(self.args.train_epochs):
            train_loss=[]
            model.train()
            for _, (inputs, labels) in enumerate(train_loader):
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad(set_to_none = True)
                preds = model(inputs)
                loss = criterion(preds,labels)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            self.record.add_outcome_to_record('epoch'+str(epoch), np.average(train_loss), self.metric(), if_print=True)
        return self

    def predict(self, pred_loader=None):
        if pred_loader is None:
            pred_loader = self.data_loader.predict 
        model =self.model
        device = self.device
        
        model.eval()
        preds = torch.tensor([])
        for _, (inputs, _) in enumerate(pred_loader):
            pred = model(inputs.to(device)).cpu().detach()
            preds = torch.concat([preds, pred], axis=0)
        return preds

    def metric(self, pred_loader=None):
        if self.args.name not in ['cifar100','cifar10','mnist']:
            return None
        if pred_loader is None:
            pred_loader = self.data_loader.predict
        y_true = pred_loader.dataset.targets
        y_pred = self.predict(pred_loader).argmax(axis=1)
        return accuracy_score(y_true, y_pred)
    
    def _acquire_device(self):
        if self.if_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda')
            print('Use GPU')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_data_loader(self):
        data_loader = ABC_Data_Loader(self.args, self.data)
        return data_loader
    
    def _build_model(self):
        if self.args.model == 'resnet':
            model = ResNet26(num_classes=10).to(self.device)
            return model
        if self.if_hash:
            self.hash = self.get_hash(self.data_loader.train.dataset.data)
        else:
            self.hash = None
        model = ABC_Net(self.args, self.hash).to(self.device)
        return model
    
    def _build_record(self):
        if self.record_path == False:
            if_save = False
        else:
            if_save = True
        if type(self.record_path) is not str:
            self.record_path = 'record/records/' + self.args.name + '/'
        record = EXPERecords(record_path=self.record_path, build_new_file=False, if_save=if_save)
        record.add_record(self.args)
        return record
    
    def _select_optimizer(self):
        if self.args.optimizer == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.0001)
        else:
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0001)
        return model_optim
    
    def _select_scheduler(self, optimizer):
        if self.args.scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        elif self.args.scheduler == 'multistep':
            scheduler = MultiStepLR(optimizer, milestones=[30,60,90,120,150,180,210,240], gamma=0.5)
        elif self.args.scheduler == 'multistep2':
            scheduler = MultiStepLR(optimizer, milestones=[30,35,40,45,50,55,60,65], gamma=0.5)
        elif self.args.scheduler == 'multistep3':
            scheduler = MultiStepLR(optimizer, milestones=[4,8,12,16,20,24], gamma=0.5)
        else:
            scheduler = MultiStepLR(optimizer, milestones=[self.args.train_epochs], gamma=1)
        return scheduler

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

    def get_hash(self, data_mat:torch.tensor):
        if self.args.name in ['cifar10', 'cifar100']:
            data_mat = torch.tensor(data_mat).permute(0,3,1,2)
        # data_mat = data_mat[:,:,0,:]
        data_shape = data_mat.shape
        # data_mat =  data_mat.reshape(data_shape[0], -1, data_shape[-2], data_shape[-1])
        data_mat =  data_mat.reshape(-1, 1, data_shape[-2], data_shape[-1])
        B,C,H,W = data_mat.shape
        hash = torch.empty((0))
        for channel in range(C):
            data_mat1 = data_mat[:,channel,:]
            _, Hi, Wi = data_mat1.shape
            data_mat1 = data_mat1.reshape(-1, Hi*Wi).T
            cov = torch.cov(data_mat1).abs()
            var = torch.var(data_mat1.to(torch.float32), axis=1).abs()
            var[var<0.01] = var[var<0.01] + 1
            corr = (cov/var.pow(0.5)).T/var.pow(0.5)
            hash = torch.concat([hash, corr.reshape(H, W, H*W).unsqueeze(0)], axis=0)
        return hash