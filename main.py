import torch
from torch import Tensor
from dataset import SequenceDataset
from torch.utils.data import DataLoader
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from utils import *


#model_type=0: non-transformer models
#model_type=1: transformer model

def train_with_test(n_epochs, model, optimiser, loss_fn, train_loader,
                  test_loader, model_type=0):
    num_train_batches = len(train_loader)
    num_test_batches = len(test_loader)
    X, tgt, y = next(iter(train_loader))
    if model_type == 1:
        src_mask = generate_square_subsequent_mask(len(y[0]), len(X[0]))
        print(len(y[0]))
        print(len(X[0]))
        tgt_mask = generate_square_subsequent_mask(len(y[0]), len(y[0]))
    for epoch in range(n_epochs):
        total_loss = 0
        total_test_loss = 0
        i = 0
        for X_train, dec_train, y_train in train_loader:
            model.train()
            if model_type == 0:
                outputs = model.forward(X_train) # forward pass
            elif model_type == 1:
                outputs = model.forward(X_train, dec_train, src_mask, tgt_mask) # forward pass through encoder/decoder model
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
        model.eval()
        for X_test, y_test in test_loader:
                test_preds = model(X_test)
                test_loss = loss_fn(test_preds, y_test)
                test_loss.backward()
                total_test_loss += test_loss.item()
        avg_loss = total_loss / num_train_batches
        avg_test_loss = total_test_loss / num_train_batches
        print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
                                                                avg_loss, 
                                                                avg_test_loss))

def train(n_epochs, model, optimiser, loss_fn, train_loader, model_type=0):

    num_batches = len(train_loader)
    i, batch = next(enumerate(train_loader))
    X, tgt, y = batch
    if model_type == 1:
        src_mask = generate_square_subsequent_mask(len(y[0]), len(X[0]))
        tgt_mask = generate_square_subsequent_mask(len(y[0]), len(y[0]))
    for epoch in range(n_epochs):
        
        total_loss = 0
        i = 0
        for X, dec, y in train_loader:
            if model_type == 0:
                outputs = model.forward(X) # forward pass
            elif model_type == 1:
                outputs = model.forward(X, dec, src_mask, tgt_mask) # forward pass through encoder/decoder model
            '''
            def closure():
                optimiser.zero_grad()
                out = model(X)
                loss = loss_fn(out, y)
                loss.backward()
                return loss
            '''
            optimiser.zero_grad() # calculate the gradient, manually setting to 0
            #  obtain the loss function
            loss = loss_fn(outputs, y)
            loss.backward() # calculates the loss of the loss function
            #optimiser.step(closure) #(use this with closure() if using LBFGS optimiser)
            optimiser.step()
            total_loss += loss.item()
            #uncomment this to see avg loss as batches go in
            '''
            current_train_avg = total_loss / (i+1)
            if i % 20 == 0:
                print("batch #: %d, current average train loss: %1.5f" % (i, current_train_avg))
            i += 1
            '''
        avg_loss = total_loss / num_batches
        print("Epoch: %d, train loss: %1.5f" % (epoch, avg_loss))

def test(model, loss_fn, test_loader, model_type=0):

    num_batches = len(test_loader)
    outputs = torch.tensor([])
    total_loss = 0
    i, batch = next(enumerate(test_loader))
    X, tgt, y = batch
    if model_type == 1:
        src_mask = generate_square_subsequent_mask(len(y[0]), len(X[0]))
        tgt_mask = generate_square_subsequent_mask(len(y[0]), len(y[0]))
    model.eval()
    with torch.no_grad():
        for X, dec, y in test_loader:
            if model_type == 0:
                y_star = model.forward(X) # forward pass
            elif model_type == 1:
                y_star = model.forward(X, dec, src_mask, tgt_mask) # forward pass through encoder/decoder model
            total_loss += loss_fn(y_star, y).item()
            outputs = torch.cat((outputs, y_star), 0)
    avg_loss = total_loss / num_batches
    print("test loss: %1.5f" %(avg_loss))
    return outputs, avg_loss

def predict(model, data_loader, model_type=0):

    #dataset = SequenceDataset(scaled_df.tail(sequence_length), target, features, sequence_length, pred_length)
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    outputs = torch.tensor([])
    i, batch = next(enumerate(data_loader))
    X, tgt, y = batch
    if model_type == 1:
        src_mask = generate_square_subsequent_mask(len(y[0]), len(X[0]))
        tgt_mask = generate_square_subsequent_mask(len(y[0]), len(y[0]))
    model.eval()
    with torch.no_grad():
        for X, dec, _ in data_loader:
            if model_type == 0:
                y_star = model.forward(X) # forward pass
            elif model_type == 1:
                y_star = model.forward(X, dec, src_mask, tgt_mask) # forward pass through encoder/decoder model
            outputs = torch.cat((outputs, y_star), 0)
    return outputs

def predict_from_array(model, data):

    #dataset = SequenceDataset(scaled_df.tail(sequence_length), target, features, sequence_length, pred_length)
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    outputs = np.empty(0,7)
    model.eval()
    with torch.no_grad():
        for i in data.shape[0]:
            pred = model(data[i]).numpy()
            outputs = np.concatenate((outputs, [pred]), axis=0)
    return outputs

#def hybrid_model_predict(model, ):


def test_results(model, scaled_train_df, scaled_test_df, target, features, show_plot=True, sequence_length=21, pred_length=7, model_type=0):
    preds = torch.tensor([])
    df_1 = scaled_train_df.copy()
    df_2 = scaled_test_df.copy()
    if model_type == 1:
        src_mask = generate_square_subsequent_mask(pred_length, sequence_length)
        tgt_mask = generate_square_subsequent_mask(pred_length, pred_length)
    model.eval()
    if model_type == 0:
        for i in range(len(df_2)):
            preds = torch.cat((preds, model(torch.tensor([df_1[features].tail(sequence_length).values], dtype=torch.float32)).detach()))
    elif model_type == 1:
        for i in range(len(df_2)):
            preds = torch.cat((preds, model(torch.tensor([df_1[features].tail(sequence_length).values], dtype=torch.float32), 
                                            torch.tensor(df_1[target].tail(1).values, dtype=torch.float32).repeat(pred_length).reshape((1,pred_length)),
                                            src_mask, tgt_mask).detach()))
    preds = torch.transpose(preds, 0, 1)
    Y_plot = ['Close']
    for i in range(pred_length):
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
    return preds

def hybrid_test_results(model, scaled_train_df, scaled_test_df, target, features, show_plot=True):
    preds = torch.tensor([])
    df_1 = scaled_train_df.copy()
    df_2 = scaled_test_df.copy()
    for i in range(len(df_2)):
        df_1['resids'] = get_residuals(df_1['Close'])
        preds = torch.cat((preds, model(torch.tensor([df_1[features].tail(21).values], dtype=torch.float32)).detach()))
        df_1.drop(columns=['resids'], inplace=True)
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
    return preds

def supervised_from_dataframe(df, target, features, sequence_length=21, pred_length=7):

    dataset = SequenceDataset(df, target, features, sequence_length=sequence_length, pred_length=pred_length)
    data_loader =  DataLoader(dataset, batch_size=None, shuffle=False)
    X_b, y_b = np.empty((0, len(features)*sequence_length)), np.empty((0, pred_length))
    for X, dec, y in data_loader:
        X_0 = np.array([])
        for i in range(len(X)):
            X_0 = np.concatenate((X_0, X[i].numpy()))
        X_b = np.concatenate((X_b, [X_0]), axis=0)
        y_b = np.concatenate((y_b, [y.numpy()]), axis=0)
    return X_b, y_b
