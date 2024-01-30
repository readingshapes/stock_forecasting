# File for Combined Plots :)

# import necessary libraries
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
import pmdarima as pm
import pickle

# format setup and suppress warnings 
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


def main():
# --------- SARIMA MODEL ----------
    apple_data = yf.download("AAPL", start = "2017-01-01", interval = '1d')
    apple_df_test = apple_data.reset_index()
    apple_df = apple_df_test.copy()


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

    #Seasonal - fit stepwise auto-ARIMA
    #!pip install pmdarima
    # Remove any duplicate index
    apple_data = apple_data.loc[~apple_data.index.duplicated(keep='first')]
    #Filter only required data
    apple_data = apple_data[['Close']]


    # scale the APPL data into a standard range using MinMaxScaler()
    Feature_Scaler = MinMaxScaler()
    # transform current APPL data
    apple_transformed = pd.DataFrame(np.squeeze(Feature_Scaler.fit_transform(apple_data), axis=1), columns=["Close"], index=apple_data.index)

    sarima_model = pm.auto_arima(apple_transformed["Close"], start_p=1, start_q=1, test='adf',
        max_p=3, max_q=3,
        m=12, #12 is the frequency of the cycle (yearly)
        start_P=0,
        seasonal=True, #set to seasonal
        d=None,
        D=1, #order of the seasonal differencing
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True)
    
'''

    # Serialize with Pickle and save it as pkl
    with open('sarima_model.pkl', 'wb') as pkl:
        pickle.dump(sarima_model, pkl)

    # Desiarilize the content of the file back into a Python object
    with open('sarima_model.pkl', 'rb') as pkl:
        loaded_model = pickle.load(pkl)

    # Call Forecast function for SARIMA model
    test = forecast(loaded_model, apple_transformed, "2023-12-20")



# --- forecast function for SARIMA model ---
def forecast(model, df, forecast_date):
    # Forecast
    n_periods = (datetime.strptime(forecast_date, '%Y-%m-%d') - df.index[-1]).days
    fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(days=1), periods = n_periods, freq='D')

    # Make series for plotting purpose
    fitted_series = pd.Series(fitted.values, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    #Concatenate the original DataFrame with forecasted values and confidence intervals
    df_result = pd.concat([df, fitted_series, lower_series, upper_series], axis=1)
    df_result.columns = ["Actual", "Prediction", "Low", "High"]

    #Inverse transform the scaled values to their original scale
    for column in df_result.columns:
      df_result[column] = Feature_Scaler.inverse_transform(df_result[column].values.reshape(-1,1))

    return df_result


'''

main()






