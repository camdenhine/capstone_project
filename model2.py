import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import zeros
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size = 5, output_size = 7, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers # number of hidden layers
        self.input_size = input_size #number of input features
        self.output_size = output_size # size of output
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, self.output_size)
        
    def forward(self, x):
        outputs, n_samples = [], x.size(0)
        h_t = zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for input_t in x.split(1,dim=1):
            # N, 1
            h_t, c_t = self.lstm1(torch.reshape(input_t,(n_samples,self.input_size)), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            #outputs.append(output)

        output = self.linear(h_t2) # output from the last FC layer
        # transform list to tensor
        #outputs = torch.cat(outputs, dim=1)
        return output