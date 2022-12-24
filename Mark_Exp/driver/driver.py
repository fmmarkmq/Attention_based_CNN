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
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.data_loader = self._build_data_loader()

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
        model = ABC_Net(self.args).to(self.device)
        return model
    
    def _build_data_loader(self):
        data_loader = ABC_data_loader(self.args)
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
        return criterion