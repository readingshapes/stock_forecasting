# Julia's source Python file :)
# !/usr/bin/python3

#1: load data into source file

# import yahoo finance data
from google.cloud import bigquery
from google.oauth2 import service_account
import yfinance as yf
# import stockstats data
from stockstats import StockDataFrame as ss

# import necessary libraries
import matplotlib as mp
import numpy as np
import pandas as pd
import pytz
import warnings
import time
import random
import statistics
import pydoc
import os
import pyarrow
import pandas_gbq
import statsmodels
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
#from tqdm.notebook import tnrange
from datetime import datetime
from datetime import timedelta
from datetime import date
import matplotlib.pyplot as graph

def main():
    # download data for apple stock
    apple_ticker = yf.Ticker("AAPL")
    # download data from past 20 years
    apple_data = yf.download("AAPL", start='2004-01-01', interval='1d')
    apple_df = ss.retype(apple_data)
    #print(apple_data)

    #2: load data into sql -> bigquery

    # import google cloud service account and bigquery
    from google.oauth2 import service_account
    from google.cloud import bigquery

    # specify google cloud project information
    credentials = service_account.Credentials.from_service_account_file('black-vehicle-406619-bf2e31773163.json')
    project_id = 'black-vehicle-406619'
    client = bigquery.Client(project=project_id, credentials=credentials)
    dataset_id = 'stocks_ds'
    table_id = '20yrs_stockdata'
    table_path = f"{project_id}.{dataset_id}.{table_id}"

    # specify load reqs
    load_info = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    load_data = client.load_table_from_dataframe(
    apple_data, table_path, job_config=load_info)
    load_data.result()

    # get number of rows in big query table
    client_table = client.dataset(dataset_id, project=project_id).table(table_id)
    number_rows = client.get_table(client_table).num_rows

    # determine slice of data for training set (~70%)
    train_vals_cutoff = int(len(apple_data) * 0.7)
        
    # determine feature length and target vals for lstm model
    time_step_vals, target_vals = features_targets(apple_data["close"].values, 20)
    # call function to create model for first test length
    scaler = MinMaxScaler(feature_range=(0,1))
    apple_df['close_price_scaled'] = scaler.fit_transform(apple_df['close'].values.reshape(-1, 1))
    model, loss, pre, predict, Y_test = create_model(time_step_vals, target_vals, apple_df["close"].values, train_vals_cutoff, scaler)
    # call function to predict lstm model
    #predict = lstm_predict(model, apple_df["close"].values, forecast_date='2024-01-03')

    
    graph.plot(apple_df["Date"][train_vals_cutoff + 20:], scaler.inverse_transform(Y_test.reshape(-1, 1)), label='Actual Prices')
    graph.plot(apple_df["Date"][train_vals_cutoff + 20:], predict, label='Predicted Prices')
    graph.xlabel('Date')
    graph.ylabel('Stock Price')
    graph.legend()
    graph.show()

#3: -- FUNCTIONS NEEDED FOR LSTM MODEL CREATION --

# create feature length list and target list - needed for lstm model
def features_targets(data, feature_length):
    # feature length is the number of time steps in the input sequence
    # targets are the values the model is trying to forecast
    time_step_list, close_label_list = [], []
    
    # iterate through (length of sequential data) to (length of seq data - feature length)
    for i in range(len(data) - feature_length):
        # this will get the vals leading up to the target
        time_steps = data[i : i + feature_length]
        time_step_list.append(time_steps)
        # this will get the target val at this point
        labels = data[i + feature_length]
        close_label_list.append(labels)

    # reshape lists to be suitable for network algo
    time_step_list = np.array(time_step_list).reshape(len(time_step_list), feature_length, 1)
    close_label_list = np.array(close_label_list).reshape(len(close_label_list), 1)

    return time_step_list, close_label_list

# -- CREATE BIDIRECTIONAL LSTM --
def create_model(X, Y, data, train_test_slice, scaler):
    # training set: set to train the machine learning model
    # testing set: set used to test model after model has been trained
    # train set is 70% of data, test set is the rest (30%)
    X_train, X_test = X[:train_test_slice], X[train_test_slice:]
    Y_train, Y_test = Y[:train_test_slice], Y[train_test_slice:]

    # initialize empty model where nodes have input and output
    model = Sequential()
    # create a bidirectional LSTM
    model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.1, input_shape=(X_train.shape[1], 1))))
    model.add(LSTM(50, recurrent_dropout=0.1))
    # add dropout and dense layers
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='elu'))
    # optimize model using stochastic gradient descent to train model
    optimize = tf.keras.optimizers.SGD(learning_rate = 0.001)
    # compile model
    #model.compile(loss='mean_squared_error', optimizer=optimize, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    model.compile(loss='mean_squared_error', optimizer=optimize)
    # save model weights when needed
    #weights = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    # adjust learning rate when needed
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=4, min_lr=0.00001,verbose = 1)
    # !!!changed epochs from 10 to 1
    # fit model
    #history = model.fit(X_train, Y_train, epochs=1, batch_size = 1, verbose=1, shuffle=False, validation_data=(X_test , Y_test), callbacks=[reduce_lr, weights])
    loss = model.evaluate(X_test, Y_test)
    print(f'Mean Squared Error on Test Set: {loss}')
    pred = model.predict(X_test)
    preds = scaler.inverse_transform(pred)
   
    return model, loss, pred, preds, Y_test
    
    
main()





















