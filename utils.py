from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

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
	'''
	scaler_open = MinMaxScaler(feature_range=(0, 1))
	scaler_high = MinMaxScaler(feature_range=(0, 1))
	scaler_low = MinMaxScaler(feature_range=(0, 1))
	scaler_close = MinMaxScaler(feature_range=(0, 1))
	scaler_volume = MinMaxScaler(feature_range=(0, 1))

	o = np.array(df['Open'])
	h = np.array(df['High'])
	l = np.array(df['Low'])
	c = np.array(df['Close'])
	v = np.array(df['Volume'])

	scaler_open.fit(o.reshape(-1,1))
	scaler_high.fit(h.reshape(-1,1))
	scaler_low.fit(l.reshape(-1,1))
	scaler_close.fit(c.reshape(-1,1))
	scaler_volume.fit(v.reshape(-1,1))

	scaled_df['Open'] = scaler_open.transform(o.reshape(-1,1))
	scaled_df['High'] = scaler_high.transform(h.reshape(-1,1))
	scaled_df['Low'] = scaler_low.transform(l.reshape(-1,1))
	scaled_df['Close'] = scaler_close.transform(c.reshape(-1,1))
	scaled_df['Volume'] = scaler_volume.transform(v.reshape(-1,1))

	return scaled_df, [scaler_open, scaler_high, scaler_low, scaler_close, scaler_volume]
	'''
def apply_scalers(df, scalers):
	
	new_df = df.copy()
	for i in range(len(scalers)):
		new_df[str(df.columns[i])] = scalers[i].transform(np.array(df.iloc[:, i]).reshape(-1,1))

	return new_df
	
	'''
	df['Open'] = scalers[0].transform(np.array(df['Open']).reshape(-1,1))
	df['High'] = scalers[1].transform(np.array(df['High']).reshape(-1,1))
	df['Low'] = scalers[2].transform(np.array(df['Low']).reshape(-1,1))
	df['Close'] = scalers[3].transform(np.array(df['Close']).reshape(-1,1))
	df['Volume'] = scalers[4].transform(np.array(df['Volume']).reshape(-1,1))
	return df
	'''

def preprocess_data(df):
	if 'date' in list(df.columns):
		df.set_index('date', inplace=True)
	df = df.reindex(columns=['Open', 'Close', 'High', 'Low', 'Volume'])

	return df

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)

	return agg.values

def get_p_value(train_series):
    # p-value>0.562 or Critical Value(1%)>-3.44, non-stationary
    t = adfuller(train_series.values)
    return t[1]
