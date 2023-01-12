import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ATD_Train_Dataset(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52, predict_len=4):
        self.df=df
        self.history_len = history_len
        self.predict_len = predict_len
        self.data = torch.tensor(df[:-self.predict_len].values).to(torch.float32).unsqueeze(dim=-2)

    def __len__(self):
        return len(self.data)-self.history_len-self.predict_len+1
    
    def __getitem__(self,idx):
        history_len = self.history_len
        predict_len = self.predict_len
        train_x = self.data[idx : idx+history_len]
        train_y = self.data[idx+history_len : idx+history_len+predict_len]
        return train_x, train_y


class ATD_Pred_Dataset(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52):
        self.df=df
        self.history_len = history_len
        self.data = torch.tensor(df[-history_len:].values).to(torch.float32).unsqueeze(dim=-2)

    def __len__(self):
        return 1

    def __getitem__(self,idx):
        history_len = self.history_len
        pred_x = self.data[idx : idx+history_len]
        return pred_x, torch.empty((0))
