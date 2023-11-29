
'''
title: stock_forecasting
author: 
    - julia
    - kari
    - kevin
start-date: 2023-09-21
description:
    - 20-years-of-data
    - lstm-models
'''

#%pip install statsmodels
#%pip install tensorflow
#%pip install yfinance
#%pip install stockstats
#%pip install pandas_gbq
#%pip install pydata_google_auth


import pandas as pd
import warnings
import pathlib
import openpyxl
from openpyxl import Workbook
import numpy as np
import collections
import sidetable
import matplotlib.pyplot as plt
import random
import datetime
import re
import base64
import pyodbc 
import sqlalchemy
import statsmodels
import yfinance as yf
import pandas_gbq
import pydata_google_auth

from stockstats import StockDataFrame as sdf
from google.cloud import bigquery


# step 1: import data
# Stock Data 
stock_data = yf.download("GOOGL" , start = "2018-01-01" , interval = '1d')
#print(stock_data.head())

stock_df = sdf.retype(stock_data)
stock_data[['stochrsi', 'macd', 'mfi']]=stock_df[['stochrsi', 'macd', 'mfi']]

print(stock_data)


# step 2: Connect to db
# big query

SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive',
]

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(
    ' ~/stock_forecasting/nifty-might-404319-b6aac4c63637.json')

df = pandas_gbq.read_gbq(
    "SELECT my_col FROM `my_dataset.my_table`",
    project_id='YOUR-PROJECT-ID',
    credentials=credentials,
)




