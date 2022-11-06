import torch
from data import SequenceDataset
from torch.utils.data import DataLoader
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from utils import *

def train_model_with_test(n_epochs, lstm, optimiser, loss_fn, train_loader,
                  test_loader):
    num_train_batches = len(train_loader)
    num_test_batches = len(test_loader)
    for epoch in range(n_epochs):
        total_loss = 0
        total_test_loss = 0
        i = 0
        for X_train, y_train in train_loader:
            lstm.train()
            outputs = lstm.forward(X_train) # forward pass
            optimiser.zero_grad() # calculate the gradient, manually setting to 0
            #  obtain the loss function
            loss = loss_fn(outputs, y_train)
            loss.backward() # calculates the loss of the loss function
            optimiser.step() # improve from loss, i.e backprop 
            total_loss += loss.item()
            current_train_avg = total_loss / (i+1)
            if i % 20 == 0:
                print("batch #: %d, current train loss: %1.5f" % (i, current_train_avg))
            i += 1

        #test loss
        lstm.eval()
        for X_test, y_test in test_loader:
                test_preds = lstm(X_test)
                test_loss = loss_fn(test_preds, y_test)
                test_loss.backward()
                total_test_loss += test_loss.item()
        avg_loss = total_loss / num_train_batches
        avg_test_loss = total_test_loss / num_train_batches
        print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
                                                                avg_loss, 
                                                                avg_test_loss))

def train_model(n_epochs, lstm, optimiser, loss_fn, train_loader):

    num_batches = len(train_loader)

    for epoch in range(n_epochs):
        
        total_loss = 0
        i = 0
        for X, y in train_loader:
            lstm.train()
            '''
            def closure():
                optimiser.zero_grad()
                out = lstm(X)
                loss = loss_fn(out, y)
                loss.backward()
                return loss
            '''
            outputs = lstm.forward(X) # forward pass
            optimiser.zero_grad() # calculate the gradient, manually setting to 0
            #  obtain the loss function
            loss = loss_fn(outputs, y)
            loss.backward() # calculates the loss of the loss function
            #o)ptimiser.step(closure # improve from loss, i.e backprop (use this with closure() if using LBFGS optimiser)
            optimiser.step()
            total_loss += loss.item()
            current_train_avg = total_loss / (i+1)

            if i % 20 == 0:
                print("batch #: %d, current average train loss: %1.5f" % (i, current_train_avg))
            i += 1
        avg_loss = total_loss / num_batches
        print("Epoch: %d, train loss: %1.5f" % (epoch, avg_loss))

def test_model(lstm, loss_fn, test_loader):

    num_batches = len(test_loader)
    total_loss = 0
    lstm.eval()
    with torch.no_grad():
        for X, y in test_loader:
            outputs = lstm(X)
            total_loss += loss_fn(outputs, y).item()

    avg_loss = total_loss / num_batches
    print("test loss: %1.5f" %(ave_loss))

def predict(lstm, data_loader):

    #dataset = SequenceDataset(scaled_df.tail(sequence_length), target, features, sequence_length, pred_length)
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    outputs = torch.tensor([])
    lstm.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = lstm(X)
            outputs = torch.cat((outputs, y_star), 0)
    return outputs

#def hybrid_lstm_predict(lstm, ):

#must take a series
def get_residuals(series):
    i=0
    while get_p_value(series) >= 0.05:
        series = series.diff()
        if i % 2 == 0:
            series = series.fillna(0.0)
        else:
            series = series.bfill()
        if i > 10:
            break
        i += 1
    model = ARIMA(endog=series.values, order=(2, i, 0)).fit()
    return model.resid

def test_results(lstm, scaled_train_df, scaled_test_df, target, features, show_plot=True):
    preds = torch.tensor([])
    df_1 = scaled_train_df.copy()
    df_2 = scaled_test_df.copy()
    for i in range(len(df_2)):
        preds = torch.cat((preds, lstm(torch.tensor([df_1[features].tail(21).values], dtype=torch.float32)).detach()))
        df_1 = pd.concat([df_1,df_2.iloc[[i]]])
    preds = torch.transpose(preds, 0, 1)
    Y_plot = ['Close']
    for i in range(7):
        new_pred = preds[i]
        if i != 0:
            new_pred = new_pred[:-i]
            for k in range(i):
                new_pred = torch.cat((torch.tensor([new_pred[0]]), new_pred))
        df_2["day " + str(i+1) + " prediction"] = new_pred.numpy()
        Y_plot.append("day " + str(i+1) + " prediction")
    if show_plot==True:
        plt.plot(df_2.index, df_2[Y_plot])
        plt.legend(Y_plot)
        plt.show()
    return df_2


