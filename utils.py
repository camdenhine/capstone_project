from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
#from dataset import SequenceDataset
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

pd.set_option('mode.chained_assignment', None)

def scale_df(df):

	scaled_df = pd.DataFrame(df.index).set_index('date')
	scalers = []
	columns = []
	for i in range(len(df.columns)):
		columns.append(np.array(df.iloc[:, i]))
		scalers.append(MinMaxScaler(feature_range=(0, 1)))
		scalers[i].fit(columns[i].reshape(-1,1))
		scaled_df[str(df.columns[i])] = scalers[i].transform(columns[i].reshape(-1,1))

	return scaled_df, scalers

def apply_scalers(df, scalers):
	
	new_df = df.copy()
	for i in range(len(scalers)):
		new_df[str(df.columns[i])] = scalers[i].transform(np.array(df.iloc[:, i]).reshape(-1,1))

	return new_df

def preprocess_data(df):
	if 'date' in list(df.columns):
		df.set_index('date', inplace=True)
	df = df.reindex(columns=['Open', 'Close', 'High', 'Low', 'Volume'])

	return df

def get_p_value(train_series):
    # p-value>0.562 or Critical Value(1%)>-3.44, non-stationary
    t = adfuller(train_series.values)
    return t[1]



#predictions should be a pytorch tensor, in the form that comes from the predict() function
def plot_test(df, predictions):
	preds = torch.transpose(predictions, 0, 1)
	df_2 = df.copy()
	Y_plot = ['Close']
	for i in range(7):
		new_pred = preds[i]
		if i != 0:
			new_pred = new_pred[:-i]
			for k in range(i):
				new_pred = torch.cat((torch.tensor([new_pred[0]]), new_pred))
		df_2["day " + str(i+1) + " prediction"] = new_pred.numpy()
		Y_plot.append("day " + str(i+1) + " prediction")

	plt.plot(df_2.index, df_2[Y_plot])
	plt.legend(Y_plot)
	plt.show()

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

#must take series of target column
def get_residuals(series, use_p_value=True):
	i=0
	if use_p_value==True:
		while get_p_value(series) >= 0.05:
			series = series.diff()
			if i % 2 == 0:
				series = series.fillna(0.0)
			else:
				series = series.bfill()
			if i > 10:
				break
			i += 1
	else:
		series = seriers.diff()
		i=1
	model = ARIMA(endog=series.values, order=(2, i, 0)).fit()
	return model.resid