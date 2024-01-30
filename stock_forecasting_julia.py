# Julia's source Python file :)
# !/usr/bin/python3

from google.cloud import bigquery
from google.oauth2 import service_account
import yfinance as yf
from stockstats import StockDataFrame as ss
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
from datetime import datetime
from datetime import timedelta
from datetime import date
import matplotlib.pyplot as graph
import time
import plotly.graph_objects as go

def main():
    # --- DOWNLOAD YAHOO FINANCE DATA ---

    # download data for apple stock
    apple_ticker = yf.Ticker("AAPL")
    # download data from past 20 years
    #apple_data = yf.download("AAPL", start='2004-01-01', interval='1d')
    apple_data = yf.download("AAPL" , start = "2004-01-01" , interval = '1d')
    apple_df_test = apple_data.reset_index()
    apple_df = ss.retype(apple_data)
    #print(apple_data)

    # --- LOAD DATA INTO BIGQUERY ---

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

    # --- CREATE LSTM MODEL ---

    # determine slice of data for training set (~70%)
    #print(apple_df_test.head())
    #print(apple_df_test.dtypes)
    #train_vals_cutoff = apple_df[apple_data.index >= '2015-12-25'].shape[0]
    train_vals_cutoff = int(len(apple_data) * 0.7)
        
    # determine feature length and target vals for lstm model
    time_step_vals, target_vals = features_targets(apple_data["close"].values, 20)
    # scale the dataset to a specific range
    scaler = MinMaxScaler(feature_range=(0, 200))
    # scale values and place in new column in df
    apple_df['close_price_scaled'] = scaler.fit_transform(apple_df['close'].values.reshape(-1, 1))
    #model, loss, pre, predict, Y_test, scaler = create_model(time_step_vals, target_vals, apple_df["close"].values, train_vals_cutoff, scaler)
    #create_model(time_step_vals, target_vals, apple_df, apple_df["close"].values, train_vals_cutoff, scaler)
    # call function to predict lstm model
    #predict = lstm_predict(model, apple_df["close"].values, forecast_date='2024-01-03')
    
    '''
    graph.xlabel('Date')
    graph.ylabel('Stock Price')
    graph.legend()
    graph.ion()
    graph.show(block=True)
    graph.savefig('predicted_stock_prices_lstm.png')
    '''

# --- FUNCTIONS NEEDED FOR LSTM MODEL CREATION ---

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

def create_model(X, Y, df, data, train_test_slice, scaler):
    # training set: set to train the machine learning model
    # testing set: set used to test model after model has been trained
    # train set is 70% of data, test set is the rest (30%)
    X_train, X_test = X[:-train_test_slice], X[-train_test_slice:]
    Y_train, Y_test = Y[:-train_test_slice], Y[-train_test_slice:]

    # shape = 20
    #print(X_train.shape[1])

    # initialize empty model where nodes have input and output with Keras
    model = Sequential()
    # create a bidirectional LSTM: 
    # - 100 cells
    # - return output for input
    # - reduce overfitting w current dropout
    # - specify # of steps for target
    model.add(Bidirectional(LSTM(500, return_sequences=True, recurrent_dropout=0.1, input_shape=(X_train.shape[1], 1))))
    # provide additional processing with undirectional layer
    model.add(LSTM(250, recurrent_dropout=0.1))
    # add dropout and dense layers
    # randomly sets 20% of inputs to 0 to prevent overfitting
    model.add(Dropout(0.2))
    # create a connected layer with 25 output units
    model.add(Dense(20, activation='elu'))
    model.add(Dropout(0.2))
    # create a connected layer with 10 output units
    model.add(Dense(10))

    # optimize model using stochastic gradient descent to train model
    # SGD 
    optimize = tf.keras.optimizers.SGD(learning_rate = 0.002)
    # compile model
    #model.compile(loss='mean_squared_error', optimizer=optimize, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]) 
    model.compile(loss='mean_squared_error', optimizer=optimize)
    # save model weights validation loss improves
    weights = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    # adjust learning rate when needed
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=4, min_lr=0.00001, verbose = 1)

    # one epoch completes when the entire training dataset is processed once by the model
    model.fit(X_train, Y_train, epochs=2, batch_size=1, verbose=1, shuffle=False, validation_data=(X_test, Y_test), callbacks=[reduce_lr, weights])
    actual = scaler.inverse_transform(Y_test)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = np.squeeze(actual, axis = 1)

    # create test df
    index_dates = data[-train_test_slice:]
    index_dates = pd.to_datetime(index_dates)
    test_df = pd.DataFrame({'Date': df.index[-train_test_slice:], 'Actual': actual, 'Predicted': predictions.flatten()})
    print(test_df)
    #return model, loss, predict, predict_inv, Y_test, scaler

    # Plotting
    graph.plot(df.index[-train_test_slice:], predictions, label="Predicted")
    graph.plot(df.index[-train_test_slice:], actual, label="Actual")
    graph.xlabel('Date')
    graph.ylabel('Stock Price')
    graph.legend()
    graph.show()
    graph.savefig('predicted_stock_prices_lstm.png')
    
     
    
    
main()




                                       
















