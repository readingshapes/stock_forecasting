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
#format setup and suppress warnings 
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

def main():
    # --- DOWNLOAD YAHOO FINANCE DATA ---

    # download data for apple stock
    #apple_ticker = yf.Ticker("AAPL")
    # download data from past 20 years
    apple_data = yf.download("AAPL", start = "2004-01-01", interval = '1d')
    apple_df_test = apple_data.reset_index()
    #apple_df = ss.retype(apple_df_test)
    apple_df = apple_df_test.copy()
    #print(apple_df.head())
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
    
    #print(train_vals_cutoff.shape)
        
    # determine feature length and target vals for lstm model
    
    # scale the dataset to a specific range
    scaler = MinMaxScaler()
    # scale values and place in new column in df
    #apple_df['close_price_scaled'] = scaler.fit_transform(apple_df['Close'].values.reshape(-1, 1))
    #print(apple_df.dtypes)
    #apple_df['close_price_scaled'] = scaler.fit_transform(apple_df["Close"].values)
    data_transformed = pd.DataFrame(
        np.squeeze(
            scaler.fit_transform(
                apple_df[["Close"]])), columns=["Close"], index=apple_df.index)

    time_step_vals, target_vals = features_targets(data_transformed["Close"].values, 20)
    
    train_vals_cutoff = apple_df.loc[apple_df['Date'] >= '2022-01-01']
    #print(train_vals_cutoff.dtypes)
    slice = train_vals_cutoff.shape[0]
    #print(data_transformed.head(25))
    model, X_train, X_test, Y_train, Y_test = create_model(
        time_step_vals, target_vals, apple_df, data_transformed["Close"].values, slice, scaler)
    # call function to predict lstm model
    #predict = lstm_predict(model, apple_df["close"].values, forecast_date='2024-01-03')

    total_x = np.concatenate((X_train, X_test), axis = 0)
    total_y = np.concatenate((Y_train, Y_test), axis = 0)
    final_predict = model.predict(total_x)
    final_predict = scaler.inverse_transform(final_predict)
    actual = scaler.inverse_transform(total_y)
    final_predict = np.squeeze(final_predict, axis = 1)
    actual = np.squeeze(actual, axis = 1)

    print(len(apple_df["Date"]))
    graph.plot(final_predict, label = "total predict")
    graph.plot(actual, label = "total actual")
    graph.xlabel('Date')
    graph.ylabel('Stock Price')
    graph.legend()
    #graph.savefig('predicted_stock_prices_lstm_total.png')
    graph.savefig('test.png')
    graph.show()

    model.predict(data_transformed[-21:-1].values.reshape(1,20,1))
    data = pd.DataFrame(apple_data)
    test = predict_lstm(model, data, '2024-02-01', scaler)
    print(test.tail())

    
    


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

    # initialize empty model where nodes have input and output with Keras
    model = Sequential()
    # create a bidirectional LSTM: 
    # - 100 cells
    # - return output for input
    # - reduce overfitting w current dropout
    # - specify # of steps for target
    model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.1, input_shape=(X_train.shape[1], 1))))
    # provide additional processing with undirectional layer
    model.add(LSTM(50, recurrent_dropout=0.1))
    # add dropout and dense layers
    # randomly sets 20% of inputs to 0 to prevent overfitting
    model.add(Dropout(0.2))
    # create a connected layer with 25 output units
    model.add(Dense(20, activation='elu'))
    model.add(Dropout(0.2))
    # create a connected layer with 10 output units
    model.add(Dense(10))
    # create a connected layer with 1 output unit
    model.add(Dense(1))

    # optimize model using stochastic gradient descent to train model
    # SGD 
    optimize = tf.keras.optimizers.SGD(learning_rate = 0.002)
    # compile model
    model.compile(loss='mean_squared_error', optimizer=optimize)
    # save model weights validation loss improves
    weights = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    # adjust learning rate when needed
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=4, min_lr=0.00001, verbose=1)

    # one epoch completes when the entire training dataset is processed once by the model
    #model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=1, shuffle=False, validation_data=(X_test, Y_test), callbacks=[reduce_lr, weights])
    actual = scaler.inverse_transform(Y_test)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = np.squeeze(actual, axis=1)
    predictions = np.squeeze(predictions, axis=1)

    #reassign index before it goes into model
    test_df = pd.DataFrame({'Actual': actual, 'Predicted': predictions.flatten()})
    print(test_df)
    #return model, loss, predict, predict_inv, Y_test, scaler

    '''
    # Plotting test set
    graph.plot(df.index[-train_test_slice:], predictions, label="Predicted")
    graph.plot(df.index[-train_test_slice:], actual, label="Actual")
    graph.xlabel('Date')
    graph.ylabel('Stock Price')
    graph.legend()
    graph.savefig('predicted_stock_prices_lstm3_test.png')
    graph.show()
    '''
    return model, X_train, X_test, Y_train, Y_test


def predict_lstm(model, df, future_date, scaler, feature_length=20):
    # iterate through today's date until future date
    for i in range((datetime.strptime(future_date, '%Y-%m-%d') - df.index[-1]).days):
        # specify close values
        feature_column = df['Close'].values
        # pick out last 20 days
        time_steps = feature_column[-feature_length:]
        # reshape array
        time_steps = time_steps.reshape(feature_length, 1)
        # scale array
        time_steps = scaler.transform(time_steps)
        # Reshape the Features array back to two dimensions
        # Predict using the model
        prediction = model.predict(time_steps.reshape(1, feature_length, 1))
        prediction = scaler.inverse_transform(prediction)
        # Create a DataFrame for the forecast and concatenate it with the original DataFrame
        df_forecast = pd.DataFrame(prediction, index=[df.index[-1] + timedelta(days=1)], columns=['Close'])
        df = pd.concat([df, df_forecast])
    return df

     
main()



                                       
















