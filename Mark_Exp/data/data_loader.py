import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
from data.timeseries_data import TimeSeries_Train_Dataset, TimeSeries_Pred_Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ABC_Data_Loader(object):
    def __init__(self, args, data=None):
        self.args = args
        self.data =data
        self.train = self.train_data_loader()
        self.predict = self.predict_data_loader()

    def train_data_loader(self):
        if self.args.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST('../../data/ABC/mnist', train=True, download=True, transform=transform)
            train = torch.utils.data.DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True)
        elif self.args.name == 'cifar10':
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])         
            dataset = datasets.CIFAR10(root='../../data/ABC/CIFAR10', train=True, download=True, transform=transform)
            train = torch.utils.data.DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=4)
        elif self.args.name == 'cifar100':
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            dataset = datasets.CIFAR100(root='../../data/ABC/CIFAR100', train=True, download=True, transform=transform)
            train = torch.utils.data.DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=4)
        elif self.args.name in ['atd', 'wiki_traffic', 'lat']:
            dataset = TimeSeries_Train_Dataset(df=self.data, history_len=self.args.history_len, predict_len=self.args.predict_len)
            train = DataLoader(dataset, batch_size = self.args.train_batch_size, shuffle=False, drop_last=False)
        return train

    def predict_data_loader(self):
        if self.args.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST('../../data/ABC/mnist', train=False, download=True, transform=transform)
            predict = torch.utils.data.DataLoader(dataset, batch_size=self.args.predict_batch_size, shuffle=False)
        elif self.args.name == "cifar10":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            testset = datasets.CIFAR10(root='../../data/ABC/CIFAR10', train=False, download=True, transform=transform)
            predict = torch.utils.data.DataLoader(testset, batch_size=self.args.predict_batch_size, shuffle=False, num_workers=4)
        elif self.args.name == "cifar100":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            testset = datasets.CIFAR100(root='../../data/ABC/CIFAR100', train=False, download=True, transform=transform)
            predict = torch.utils.data.DataLoader(testset, batch_size=self.args.predict_batch_size, shuffle=False, num_workers=4)
        elif self.args.name in ['atd', 'wiki_traffic', 'lat']:
            dataset = TimeSeries_Pred_Dataset(df=self.data, history_len=self.args.history_len)
            predict = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        return predict