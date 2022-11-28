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
        return self.X.shape[0] - self.pred_length - self.sequence_length

    def __getitem__(self, i): 
        i_start = i
        x = self.X[i_start:(i + self.sequence_length), :]
        
        #padding = self.Y[-1].repeat(self.pred_length)
        #y = torch.cat((self.Y,padding), 0)
        i_end = i + self.pred_length + 1


        return x, self.Y[i + self.sequence_length:i_end-1+self.sequence_length], self.Y[i+1+self.sequence_length:i_end+self.sequence_length]

class PredDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=21, pred_length=7):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.Y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length

    def __getitem__(self, i): 
        i_start = i + 1
        x = self.X[i_start:(i + 1 + self.sequence_length), :]
        
        padding = self.Y[-1].repeat(self.pred_length)
        y = torch.cat((self.Y,padding), 0)
        i_end = i + self.pred_length + 1


        return x, y[i + self.sequence_length:i_end-1+self.sequence_length], y[i+1+self.sequence_length:i_end+self.sequence_length]