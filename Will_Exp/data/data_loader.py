import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ABC_data_loader(object):
    def __init__(self, args):
        self.args = args
        self.train = self.train_data_loader()
        self.predict = self.predict_data_loader()

    def train_data_loader(self):
        if self.args.data.name == 'mnist':
            train = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../../data/ABC/mnist', train=True, download=True, 
                                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                                batch_size=self.args.data.train.batch_size, shuffle=True)
            return train

    def predict_data_loader(self):
        if self.args.data.name == 'mnist':
            predict = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../../data/ABC/mnist', train=False, download=True,
                                                  transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                                  batch_size=self.args.data.predict.batch_size, shuffle=False)
            return predict