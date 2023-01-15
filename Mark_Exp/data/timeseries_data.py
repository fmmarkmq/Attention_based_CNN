import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeries_Train_Dataset(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52, predict_len=4):
        self.df=df
        self.history_len = history_len
        self.predict_len = predict_len
        # self.data = torch.tensor(df.values).to(torch.float32).unsqueeze(dim=-2)
        self.data_x, self.data_y = self._build_data(df)
        self.data = self.data_x


    def __len__(self):
        return len(self.df)-self.history_len-self.predict_len+1
    
    def __getitem__(self,idx):
        train_x = self.data_x[idx]
        train_y = self.data_y[idx]
        return train_x, train_y

    def _build_data(self, df):
        data = torch.tensor(df.values).to(torch.float32)
        data_x = torch.empty((0))
        data_y = torch.empty((0))
        for i in range(self.__len__()):
            train_x = data[i : i+self.history_len]
            train_y = data[i+self.history_len : i+self.history_len+self.predict_len]
            data_x = torch.concat([data_x, train_x.unsqueeze(0).unsqueeze(1)])
            data_y = torch.concat([data_y, train_y.unsqueeze(0).unsqueeze(1)])
        return data_x, data_y


class TimeSeries_Pred_Dataset(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52):
        self.df=df
        self.history_len = history_len
        # self.data = torch.tensor(df[-history_len:].values).to(torch.float32).unsqueeze(dim=-2)
        self.data = self._build_data(df)

    def __len__(self):
        return 1

    def __getitem__(self,idx):
        pred_x = self.data[idx]
        return pred_x, torch.empty((0))
    
    def _build_data(self, df):
        data = torch.tensor(df[-self.history_len:].values).to(torch.float32)
        data_x = torch.empty((0))
        for i in range(self.__len__()):
            train_x = data[i : i+self.history_len]
            data_x = torch.concat([data_x, train_x.unsqueeze(0).unsqueeze(1)])
        return data_x
