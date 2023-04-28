import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
import torch.nn.functional as F
import torchvision
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from model.model import ABC_Net, PipelineParallelABC_Net, ResMLPNet, ConvMixer
from data.data_loader import ABC_Data_Loader
from record.record import EXPERecords
from attack.attack import Attack


class ABC_Driver(object):
    def __init__(self, args, data=None, record_path=None, if_hash=False):
        self.args = args
        self.data = data
        self.record_path = record_path
        self.if_hash = if_hash
        self.device = self._acquire_device()
        self.data_loader = self._build_data_loader()
        self.model = self._build_model()
        self.record = self._build_record()
        self.attack = self._build_attack()

    def train(self, train_loader=None):
        if train_loader is None:
            train_loader = self.data_loader.train 
        model =self.model
        device = self.device[0]
        criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
            

        for epoch in range(self.args.train_epochs):
            train_loss=[]
            model.train()
            for _, (inputs, labels) in enumerate(train_loader):
                inputs=inputs.to(device).to(torch.float16)
                labels=labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                preds = model(inputs).to(torch.float32)
                loss = criterion(preds,labels)
                train_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if self.scheduler_step_after_batch:
                    scheduler.step()
            if not self.scheduler_step_after_batch:
                scheduler.step()
            self.record.add_train_log('epoch'+str(epoch), np.average(train_loss), self.metric(), if_print=True)
        self.record.add_test_outcome(self.metric(test_attack=True))
        return self

    def predict(self, pred_loader=None):
        pred_loader = pred_loader or self.data_loader.predict 
        model =self.model
        device = self.device[0]

        model.eval()
        with torch.no_grad():
            preds = torch.tensor([])
            for _, (inputs, _) in enumerate(pred_loader):
                pred = model(inputs.to(device).to(torch.float16)).cpu().detach()
                preds = torch.concat([preds, pred], axis=0)
        return preds

    def metric(self, test_attack=False):
        if self.args.name not in ['cifar100','cifar10','mnist']:
            return None
        y_true = self.data_loader.predict.dataset.targets
        y_pred = self.predict(self.data_loader.predict).argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        if test_attack:
            metric = pd.Series({'clean':accuracy})
            for attacker_name in self.args.attack:
                metric[attacker_name] = self.attack(attacker_name, self.model)
            return metric
        else:
            return accuracy
    
    def _acquire_device(self):
        if isinstance(self.args.device, str):
            device = [torch.device(self.args.device)]
        else:
            device = [torch.device(device_name) for device_name in self.args.device]
        print(f'Use: {self.args.device}')
        return device
    
    def _build_data_loader(self):
        data_loader = ABC_Data_Loader(self.args, self.data)
        return data_loader
    
    def _build_model(self):
        if self.args.model == 'resnet':
            model = torchvision.models.resnet34(num_classes=100).to(self.device[0])
            return model
        if self.args.model == 'resmlp':
            model = ResMLPNet().to(self.device[0])
            return model
        if self.args.model == 'convmixer':
            model = ConvMixer(256,16,8,1,10).to('cuda').to(self.device[0])
            return model
        if self.if_hash:
            self.hash = self.get_hash(self.data_loader.train.dataset.data)
        else:
            self.hash = None
        if len(self.device) == 1:
            model = ABC_Net(self.args, self.hash).to(self.device[0])
        else:
            model = PipelineParallelABC_Net(self.args, self.hash, self.device)
        return model.to(torch.float16)

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
    
    def _build_attack(self):
        bounds, preprocessing, attack_data_loader = self.data_loader.attack
        attack = Attack(self.args, bounds, preprocessing, attack_data_loader, device=self.device[0])
        return attack

    def _select_optimizer(self):
        if self.args.optimizer == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.0001789)
        elif self.args.optimizer == 'AdamW':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0, eps=0.001)
        else:
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0001)
        return model_optim
    
    def _select_scheduler(self, optimizer):
        self.scheduler_step_after_batch = False
        if self.args.scheduler == 'OneCycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, div_factor=25,
                                                        steps_per_epoch=len(self.data_loader.train), 
                                                        epochs=self.args.train_epochs)
            self.scheduler_step_after_batch = True
        elif self.args.scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        elif self.args.scheduler == 'multistep':
            scheduler = MultiStepLR(optimizer, milestones=[30,60,90,120,150,180,210,240], gamma=0.5)
        elif self.args.scheduler == 'multistep2':
            scheduler = MultiStepLR(optimizer, milestones=[30,35,40,45,50,55,60,65], gamma=0.5)
        elif self.args.scheduler == 'multistep3':
            scheduler = MultiStepLR(optimizer, milestones=[4,8,12,16,20,24], gamma=0.5)
        elif self.args.scheduler == 'multistep4':
            scheduler = MultiStepLR(optimizer, milestones=[81,122], gamma=0.1)
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