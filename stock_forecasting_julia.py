# Julia's source Python file :)
# !/usr/bin/python3


#1: load data into source file


# run in command line for project to be recognized: 
# export GOOGLE_APPLICATION_CREDENTIALS="/Users/juliawilliams/Projects/stock_forecasting-1/black-vehicle-406619-bf2e31773163.json"
# /opt/homebrew/bin/python3.9 /Users/juliawilliams/Projects/stock_forecasting-1/stock_forecasting_julia.py

# run in command line for installation:
# /usr/bin/python3 -m pip install pandas-gbq

# import yahoo finance data
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
import mysql.connector
import os
import pyarrow
import pandas_gbq
import statsmodels
import tensorflow

# localize time - ambiguous error
#tz = pd.Timestamp('2023-11-05')
#tz.tz_localize(pytz.timezone('US/Pacific'))

# download data for apple stock
apple_ticker = yf.Ticker("AAPL")
# download data from past 20 years
apple_data = yf.download("AAPL", start = '2004-01-01', interval = '1d')
apple_df = ss.retype(apple_data)
# introduce stochrsi, macd, mfi analyses
apple_data[['stochrsi', 'macd', 'mfi']] = apple_df[['stochrsi', 'macd', 'mfi']]
#print(apple_data)

print(apple_data)



#2: load data into sql -> bigquery


SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive',
]

# import google cloud service account and bigquery
from google.oauth2 import service_account
from google.cloud import bigquery

# specify google cloud project information
credentials = service_account.Credentials.from_service_account_file(
    'black-vehicle-406619-bf2e31773163.json')
project_id = 'black-vehicle-406619'
client = bigquery.Client(project=project_id, credentials=credentials)
dataset_id = 'stocks_ds'
table_id = '20yrs_stockdata'
table_path = f"{project_id}.{dataset_id}.{table_id}"

# specify load reqs
load_info = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
load_data = client.load_table_from_dataframe(apple_data, table_path, job_config=load_info)
load_data.result()


#3: Process the data


#from tensorflow import

#sarima = sarima_model.predict(n_periods=14, return_conf_int=True)
#lstm = lstm_model.predict(apple_data)

#forecast_sarima_and_lstm = forecast_sarima * 0.7 + forecast_lstm * 0.3



























