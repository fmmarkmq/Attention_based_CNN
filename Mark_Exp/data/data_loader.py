import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
from data.atd_data import ATD_Train_Dataset, ATD_Pred_Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ABC_Data_Loader(object):
    def __init__(self, args, data=None):
        self.args = args
        self.data =data
        self.train = self.train_data_loader()
        self.predict = self.predict_data_loader()

    def train_data_loader(self):
        if self.args.name == 'mnist':
            dataset = torchvision.datasets.MNIST('../../data/ABC/mnist', train=True, download=True, 
                                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
            train = torch.utils.data.DataLoader(dataset, batch_size=self.args.train.batch_size, shuffle=True)
        elif self.args.name == 'atd':
            dataset = ATD_Train_Dataset(df=self.data, history_len=self.args.history_len, predict_len=self.args.predict_len)
            train = DataLoader(dataset, batch_size = self.args.train.batch_size, shuffle=False, drop_last=False)
        return train

    def predict_data_loader(self):
        if self.args.name == 'mnist':
            predict = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../../data/ABC/mnist', train=False, download=True,
                                                  transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                                  batch_size=self.args.predict.batch_size, shuffle=False)
        elif self.args.name == 'atd':
            dataset = ATD_Pred_Dataset(df=self.data, history_len=self.args.history_len)
            predict = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        return predict