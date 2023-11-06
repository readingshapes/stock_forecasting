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

# temporary: working with info
#print(apple_ticker.balance_sheet)
#print(apple_data)


'''
2: load data into sql -> bigquery
'''

# establish sql connection
sql_db = mysql.connector.connect (
   # host
   # user
   # password

)

# import bigquery + service account
from google.cloud import bigquery
from google.oauth2 import service_account

# connect to bigquery
#service_acc = "cloud-workflow-sa-25545f81@nifty-might-404319.iam.gserviceaccount.com"
creds = service_account.Credentials.from_service_account_file('nifty-might-404319-b6aac4c63637.json')
project_id = 'nifty-might-404319'
client = bigquery.Client(credentials=creds, project=project_id)














