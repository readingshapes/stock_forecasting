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
from tqdm.notebook import tnrange
from datetime import datetime
from datetime import timedelta
from datetime import date


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

    # slice time: using data from 2022 to present
    test_length2 = apple_data[apple_df.index >= '2022-01-01'].shape[0]
        
    # determine feature length and target vals for lstm model
    feature_len_list, target_val_list = features_targets(apple_data["close"].values, 10)
    # call function to create model for first test length
    model = create_model(feature_len_list, target_val_list, apple_df["close"].values, test_length2)
    # call function to predict lstm model
    predict = lstm_predict(model, apple_df["close"].values, forecast_date='2024-01-03')

#3: -- FUNCTIONS NEEDED FOR LSTM MODEL CREATION --

# create feature length list and target list - needed for lstm model
def features_targets(data, feature_length):
    # feature length is the number of time steps in the input sequence
    feature_list = []
    # targets are the values the model is trying to forecast
    target_list = []

    # iterate through (length of sequential data) to (length of seq data - feature length)
    for i in tnrange(len(data) - feature_length):
        feature_list.append(data[i : i + feature_length])
        target_list.append(data[i + feature_length])

    # feature length list
    feature_list = np.array(feature_list).reshape(len(feature_list), feature_length, 1)
    # target list
    target_list = np.array(target_list).reshape(len(target_list), 1)

    return feature_list, target_list

def create_model(X, Y, data, test_length):
    # initialize empty model where nodes have input and output
    model = Sequential()
    # create a bidirectional LSTM
    model.add(Bidirectional(LSTM(500 ,return_sequences=True , recurrent_dropout=0.1, input_shape=(20, 1))))
    model.add(LSTM(250 ,recurrent_dropout=0.1))
    # add dropout and dense layers
    model.add(Dropout(0.2))
    model.add(Dense(60 , activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(30 , activation='elu'))
    model.add(Dense(1 , activation='linear'))
    # optimize model using stochastic gradient descent to train model
    optimize = tf.keras.optimizers.SGD(learning_rate = 0.001)
    # compile model
    model.compile(loss='mse', optimizer=optimize, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    # separate data into testing and training sets
    Xtrain, Xtest, Ytrain, Ytest = X[:-test_length], X[-test_length:], Y[:-test_length], Y[-test_length:]
    # save model weights when needed
    weights = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    # adjust learning rate when needed
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=4, min_lr=0.00001,verbose = 1)
    # !!!changed epochs from 10 to 1
    # fit model
    history = model.fit(Xtrain, Ytrain, epochs=1, batch_size = 1, verbose=1, shuffle=False, validation_data=(Xtest , Ytest), callbacks=[reduce_lr, weights])
    return model
    
def lstm_predict(model, df, forecast_date, feature_length=20):
    for i in range((datetime.strptime(forecast_date, '%Y-%m-%d') - df.index[-1]).days):
        Features = df.iloc[-20:].values.reshape(-1, 1)
        Features = Feature_Scaler.transform(Features)
        Prediction = Model.predict(Features.reshape(1,20,1))
        Prediction = Feature_Scaler.inverse_transform(Prediction)
        df_forecast = pd.DataFrame(Prediction, index=[df.index[-1]+ timedelta(days=1)], columns=['Close'])
        df = pd.concat([df, df_forecast])
    return df
    


main()





















