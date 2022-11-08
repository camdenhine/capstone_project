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
        if i > len(self) - 1:
            x, y = self[i - len(self)]
            return x, y
        if i < 0:
            x, y = self[len(self) + i]
            return x, y
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


        return x, y[i:i_end-1], y[i+1:i_end]

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

class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    def __init__(self, 
        data: torch.tensor,
        indices: list, 
        enc_seq_len: int = 21, 
        dec_seq_len: int = 7, 
        target_seq_len: int = 7
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence 
                        before any slicing. If univariate, data.size() will be 
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence. 
                     The sub-sequence is split into src, trg and trg_y later.  

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """
        
        super().__init__()

        self.indices = indices

        self.data = data

        print("From get_src_trg: data size = {}".format(data.size()))

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len



    def __len__(self):
        
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]

        #print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )

        return src, trg, trg_y
    
    def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence. 

        Args:

            sequence: tensor, a 1D tensor of length n where 
                    n = encoder input length + target sequence length  

            enc_seq_len: int, the desired length of the input to the transformer encoder

            target_seq_len: int, the desired length of the target sequence (the 
                            one against which the model output is compared)

        Return: 

            src: tensor, 1D, used as input to the transformer model

            trg: tensor, 1D, used as input to the transformer model

            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss. 
        
        """
        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"
        
        # encoder input
        src = sequence[:enc_seq_len] 
        
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len-1:len(sequence)-1]
        
        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] 