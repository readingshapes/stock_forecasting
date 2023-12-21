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
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def main():
    # download data for apple stock
    apple_ticker = yf.Ticker("AAPL")
    # download data from past 20 years
    apple_data = yf.download("AAPL", start='2004-01-01', interval='1d')
    apple_df = ss.retype(apple_data)
    # introduce stochrsi, macd, mfi analyses
    apple_data[['stochrsi', 'macd', 'mfi']] = apple_df[['stochrsi', 'macd', 'mfi']]
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

    test_length = apple_data["close"].values[(apple_data["close"].values.index >= '2004-01-01')].shape[0]

    # determine feature length and target vals for lstm model
    feature_len_list, target_val_list = features_targets(apple_data["close"].values, number_rows)
    model(feature_len_list, target_val_list, apple_data["close"].values, test_length)


#3: -- FUNCTIONS NEEDED FOR LSTM MODEL CREATION --

# create feature length list and target list - needed for lstm model
def features_targets(data, feature_length):
    # feature length is the number of time steps in the input sequence
    feature_list = []
    # targets are the values the model is trying to forecast
    target_list = []

    # iterate through (length of sequential data) to (length of seq data - feature length)
    for i in range(len(data) - feature_length):
        feature_list.append(data[i : i + feature_length])
        target_list.append(data[i + feature_length])

    # feature length list
    feature_list = np.array(feature_list).reshape(len(feature_list), feature_length, 1)
    # target list
    target_list = np.array(target_list).reshape(len(target_list), 1)

    return feature_list, target_list

def model(X, Y, data, test_length):
    model = Sequential()
    model.add(Bidirectional(LSTM(512 ,return_sequences=True , recurrent_dropout=0.1, input_shape=(20, 1))))
    model.add(LSTM(256 ,recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(64 , activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(32 , activation='elu'))
    model.add(Dense(1 , activation='linear'))
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002)
    model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    #index = np.where(data >= '2004-01-01')
    #test_length = data[index].shape(0)
    #test_length = data[(data.index >= '2004-01-01')].shape[0]
    Xtrain, Xtest, Ytrain, Ytest = X[:-test_length], X[-test_length:], Y[:-test_length], Y[-test_length:]
    save_best = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=4, min_lr=0.00001,verbose = 1)
    history = model.fit(Xtrain, Ytrain, epochs=10, batch_size = 1, verbose=1, shuffle=False, validation_data=(Xtest , Ytest), callbacks=[reduce_lr, save_best])



def lstm_predict():
    return None
    


main()





















