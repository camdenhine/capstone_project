import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
from utils import get_residuals

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=21, pred_length=7):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.Y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        padding = self.Y[-1].repeat(self.pred_length)
        y = torch.cat((self.Y,padding), 0)
        i_end = i + self.pred_length + 1

        return x , y[i:i_ end-1], y[i+1:i_end]

class ResidualSequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=21, pred_length=7):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.dataframe = dataframe
        self.Y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe.values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i > len(self) - 1:
            x, y = self[i - len(self)]
            return x, y
        if i < 0:
            x, y = self[len(self) + i]
            return x, y
        if i > 50:
            df = self.dataframe[:i+1].copy()
        else:
            df = self.dataframe[:51].copy()
        df['residuals'] = get_residuals(df['Close'])
        X_r = torch.tensor(df[self.features].values).float()
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = X_r[i_start:(i + 1), :]
        else:
            padding = X_r[0].repeat(self.sequence_length - i - 1, 1)
            x = X_r[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        padding = self.Y[-1].repeat(self.pred_length)
        y = torch.cat((self.Y,padding), 0)
        i_end = i + self.pred_length + 1
        del df

        return x, y[i:i_end-1], y[i+1:i_end]