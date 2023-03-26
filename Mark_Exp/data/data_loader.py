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
        self.attack = self.attack_data()

    def train_data_loader(self):
        if self.args.name == 'mnist':
            transform = transforms.Compose([transforms.RandomRotation(20),
                                            transforms.RandomAffine(0, translate=(0.2, 0.2)),
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST('../../data/ABC/mnist', train=True, download=True, transform=transform)
            train = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True)
        elif self.args.name == 'cifar10':
            transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0), ratio=[0.75, 4/3]),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandAugment(num_ops=2, magnitude=9),
                                            transforms.ColorJitter(0.4,0.4,0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            transforms.RandomErasing(p=0.25),])
            dataset = datasets.CIFAR10(root='../../data/ABC/CIFAR10', train=True, download=True, transform=transform)
            train = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=2)
        elif self.args.name == 'cifar100':
            transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0), ratio=[0.75, 4/3]),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandAugment(num_ops=2, magnitude=9),
                                            transforms.ColorJitter(0.4,0.4,0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
                                            transforms.RandomErasing(p=0.25),])
            dataset = datasets.CIFAR100(root='../../data/ABC/CIFAR100', train=True, download=True, transform=transform)
            train = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=2)
        elif self.args.name in ['atd', 'wiki_traffic', 'lat']:
            dataset = TimeSeries_Train_Dataset(df=self.data, history_len=self.args.history_len, predict_len=self.args.predict_len)
            train = DataLoader(dataset, batch_size = self.args.train_batch_size, shuffle=False, drop_last=False)
        return train

    def predict_data_loader(self):
        if self.args.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST('../../data/ABC/mnist', train=False, download=True, transform=transform)
            predict = DataLoader(dataset, batch_size=self.args.predict_batch_size, shuffle=False)
        elif self.args.name == "cifar10":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            testset = datasets.CIFAR10(root='../../data/ABC/CIFAR10', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=self.args.predict_batch_size, shuffle=False, num_workers=2)
        elif self.args.name == "cifar100":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),])
            testset = datasets.CIFAR100(root='../../data/ABC/CIFAR100', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=self.args.predict_batch_size, shuffle=False, num_workers=2)
        elif self.args.name in ['atd', 'wiki_traffic', 'lat']:
            dataset = TimeSeries_Pred_Dataset(df=self.data, history_len=self.args.history_len)
            predict = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        return predict
    
    def attack_data(self):
        if self.args.name == 'mnist':
            bounds = (0,1)
            preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = datasets.MNIST('../../data/ABC/mnist', train=False, download=True, transform=transform)
            attack = DataLoader(dataset, batch_size=self.args.predict_batch_size, shuffle=False)
        elif self.args.name == "cifar10":
            bounds = (0,1)
            preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
            transform = transforms.Compose([transforms.ToTensor()])
            testset = datasets.CIFAR10(root='../../data/ABC/CIFAR10', train=False, download=True, transform=transform)
            attack = DataLoader(testset, batch_size=self.args.predict_batch_size, shuffle=False, num_workers=2)
        elif self.args.name == "cifar100":
            bounds = (0,1)
            preprocessing = dict(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761], axis=-3)
            transform = transforms.Compose([transforms.ToTensor()])
            testset = datasets.CIFAR100(root='../../data/ABC/CIFAR100', train=False, download=True, transform=transform)
            attack = DataLoader(testset, batch_size=self.args.predict_batch_size, shuffle=False, num_workers=2)
        elif self.args.name in ['atd', 'wiki_traffic', 'lat']:
            bounds, preprocessing = None, None
            dataset = TimeSeries_Pred_Dataset(df=self.data, history_len=self.args.history_len)
            attack = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        return bounds, preprocessing, attack