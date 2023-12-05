# Julia's source Python file :)

'''
1: load data into source file
'''

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
#from pandas import pandas_gbq
from pandas.io import gbq

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
print(apple_data)
print(apple_ticker.get_capital_gains)

# temporary: working with info
#print(apple_ticker.balance_sheet)
#print(apple_data)

'''
2: load data into sql -> bigquery
'''

SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive',
]

from google.oauth2 import service_account
from google.cloud import bigquery
credentials = service_account.Credentials.from_service_account_file(
    'black-vehicle-406619-bf2e31773163.json')


df = pd.read_gbq(
    "SELECT my_col FROM `stock_ds.20yrs_stockdata`",
    project_id='black-vehicle-406619',
    credentials=credentials,
)

project_id = 'black-vehicle-406619'
dataset = 'stock_ds'
table_id = '20yrs_stockdata'
client = bigquery.Client(credentials=credentials, project=project_id)

if df.empty:
   job = client.load_table_from_dataframe(apple_data, table_id)
   job.result()
   print("There are {0} rows added/changed".format(len(apple_data)))
else:
   changes = apple_data[~apple_data.apply(tuple, 1).isin(df.apply(tuple, 1))]
   job = client.load_table_from_dataframe(changes, table_id)
   job.result()
   print("There are {0} rows added/changed".format(len(changes)))


'''
# import bigquery + service account
from google.cloud import bigquery
from google.oauth2 import service_account

# connect to bigquery
#service_acc = "cloud-workflow-sa-25545f81@nifty-might-404319.iam.gserviceaccount.com"
creds = service_account.Credentials.from_service_account_file('nifty-might-404319-b6aac4c63637.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'nifty-might-404319-b6aac4c63637.json'
project_id = 'nifty-might-404319'
client = bigquery.Client(credentials=creds, project=project_id)
table_id = 'nifty-might-404319.
'''

















