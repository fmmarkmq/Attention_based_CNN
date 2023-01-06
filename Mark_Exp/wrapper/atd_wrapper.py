import atd2022
import torch
import numpy as np
import pandas as pd
# from driver.atd_CNN_Transformer import ATD_CNN_Transformer
from driver.driver import ABC_Driver
from utils.tools import dotdict

class ABC_Forecaster():

    def __init__(self, args: dotdict):
        self.args = args

    def fit(self, df:pd.DataFrame, past_covariates=None) -> "ABC_Forecaster":
        self.df=df
        self.training = pd.DataFrame(self._fit_processing(df), columns=df.columns, index=df.index)
        exp = ABC_Driver(self.args, self.training)
        exp.train()
        self.model = exp
        return self

    def predict(self, indicies):
        predictions = self.generate_pred(indicies)
        predictions_df = self._predict_processing(predictions.values)
        return predictions_df.set_index(indicies)

    def generate_pred(self, indicies):
        model = self.model
        if "timeStamps" in self.df.columns:
            self.df = self.df.drop(["timeStamps"], axis=1)
        new_rows = model.predict().squeeze(0).squeeze(1).cpu().detach().numpy()
        new_df = pd.DataFrame(new_rows, index=indicies, columns=self.df.columns)
        return new_df

    def _fit_processing(self, data):
        if self.args.if_filter_constant:
            self.constant_columns = data.diff().loc[:, (data.diff().iloc[1:] == 0).all()].columns
            # data = data.drop(self.constant_columns, axis=1)
            self.not_constant_columns = data.columns
        if self.args.if_normalize:
            data_proc, self.mean, self.std = self._normalize_data(data)
        else:
            data_proc = np.array(data)
        return data_proc

    def _predict_processing(self, data):
        if self.args.if_normalize:
            data_back = self._verse_normalize_data(data,  self.mean, self.std)
        else:
            data_back = data

        data_back = np.round(data_back)
        data_back[data_back < 0] = 0

        if self.args.if_filter_constant:
            data_back_df = pd.DataFrame(data_back, columns=self.df.columns)
            data_back_df[self.constant_columns] = self.df.iloc[-self.args.predict_len:][self.constant_columns].values
            return data_back_df
        return pd.DataFrame(data_back, columns=self.df.columns)

    def _normalize_data(seft, data):
        data = np.array(data)
        data_mean = np.average(data, axis=0)
        # data_mean = 0
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std

    def _verse_normalize_data(self, data, data_mean, data_std):
        return data * data_std + data_mean