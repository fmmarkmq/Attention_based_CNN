import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
import re

class EXPERecords(object):
    def __init__(self, record_path, build_new_file=False): 
        self.path, self.file = self._build_file_and_path(record_path, build_new_file)
        self.record = None

    def add_record(self, experiment_args):
        self.record_index = datetime.now().strftime("%m/%d/%Y %H:%M")
        self.record = pd.Series(experiment_args.copy(), name=self.record_index)
        columns = self.record.index.union(self.file.columns, sort=False)
        self.file = pd.concat([self.file.T, self.record], axis=1).T[columns]
        print(f'add record: {self.record_index}')


    def add_outcome_to_record(self, epoch: str, train_loss, metric, if_print=True):
        if 'train_loss' not in self.record.index:
                self.record['train_loss'] = dict()
        self.record['train_loss'][epoch] = train_loss
        self.record[epoch] = metric
        self.file = pd.concat([self.file.iloc[:-1].T, self.record], axis=1).T
        self.file.to_csv(self.path, index=True)
        if if_print:
            print(f'epoch: {epoch[5:]}, train_loss: {round(train_loss, 4)}, test_metric: {metric}')
            

    def _build_file_and_path(self, record_path:str, build_new_file):
        if os.path.isfile(record_path):
            if build_new_file:
                n = self._find_next_number_str(record_path[:-record_path[::-1].find('/')], build_new_file)
                record_path_split = record_path.split('.')
                assert len(record_path_split) == 2
                path = record_path_split[0] + n + '.' + record_path_split[1]
                file = pd.DataFrame()
            else:
                path = record_path
                file = pd.read_csv(path, index_col=0)
        else:
            if '.' not in record_path:
                filename = 'record'
                if record_path[-1] != '/':
                    record_path = record_path + '/'
            else:
                filename = record_path[-record_path[::-1].find('/'):]
                record_path = record_path[:-record_path[::-1].find('/')]
            if not os.path.isdir(record_path):
                os.makedirs(record_path)
            n = self._find_next_number_str(record_path, build_new_file)
            path = record_path + filename + n +'.csv'
            if os.path.isfile(path):
                file = pd.read_csv(path, index_col=0)
            else:
                file = pd.DataFrame()
        return path, file

    
    def _find_next_number_str(self, dir_path, build_new_file):
        filenames = next(os.walk(dir_path), (None, None, []))[2]
        if len(filenames) == 0:
            return ''
        else:
            patten = re.compile('[0-9]+\.')
            numbers =[int(patten.findall(filename)[0][:-1]) for filename in filenames if len(patten.findall(filename)) > 0]
            if len(numbers) == 0:
                if build_new_file:
                    return '1'
                else:
                    return ''
            else:
                if build_new_file:
                    return str(max(numbers) + 1)
                else:
                    return str(max(numbers))

