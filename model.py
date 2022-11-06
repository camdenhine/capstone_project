import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import zeros
import numpy as np


class LSTM(nn.Module):
    
    def __init__(self, num_outputs, num_features, hidden_size, num_layers):
        super().__init__()
        self.num_outputs = num_outputs # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.num_features = num_features # inputs features
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_outputs) # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc_2(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
        '''
    def forward(self,x):
        # hidden state
        h_0 = Variable(zeros(self.num_layers, x.size(0), self.hidden_size))
        # cell state
        c_0 = Variable(zeros(self.num_layers, x.size(0), self.hidden_size))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out
        '''