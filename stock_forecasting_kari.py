
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

{
    "type": "service_account",
    "project_id": "nifty-might-404319",
    "private_key_id": "b6aac4c6363778450772da5eaa37e74a7baf918d",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC2xp6Smyl2ivqW\nrNL5eB4Z8wRQB1xrSSkfIhX1f8U00BmsgZgJJ59Cht7Ya+gw/4bOYb1Va1srJCep\nVUqEErF2zqdrHBc0kni36WtJzuY6ka6GAgP3wJmpw4Ml7kwaM4ywVT4S7oxF5L2y\n2sTRQScdZxC4B7AO1qtbmuQXeDXQLgRgKKdUXCw0lqXkGNe/WTgC4qpNj/vGE8OC\n7SYVG+m6eavWR/6uEI6VuKfGnfkYq5rNZCxAAkGdSnlX73a5p/Wvodc7ZR6CKo3u\nDf/YPLzqO0I42Smvwlij/41VozzlDlplWes6z2db3AM1ZTZiq+oqzb8l7C3lOXrp\n9eRv/Z5VAgMBAAECggEAHXWH63NMzIl7+Dskyga9O0t6/3cgQz6IfTceOPJ+E5QS\n/0Xn/lm/hpZ8Zn+F7hfRX4RLYvApwptSNS3FE+J7bf3C6DWf295bzLC3lS7e1sPS\nUFEU2KLXYZBcnAl7hKGYZHdoyN5gB4flt2UhYeTRbCDHhhHfI5UgC2S8rLe4XLQ4\nMymwstFi5BpzS7O6AYNE3C4EQ511G+WJMpYfKEG8wyga3cZu83+2qvCVqDnyzgcN\nbaSHaj/FOD9RywEcYjyik3U/mQqovEpY3/kMO3KMyAXD7k0SLLhUoafMV/kghFsD\nnKq2oF9YZl80LN8NmtJTnsvglO3blzk5hgVMB/iyoQKBgQDk3O1UhXSq6kqyBG4o\nZl2FpTAxs+u9Nr7hKWrIhJORueMOJPfsg22YBjRf+tBZ08kkuIdt755qqqNp90F9\nhDdF1PvNW8UsdHEvb/br6j4Xom77YcIN/CdpCeN9sT0umbPdD6T3Unkw+k2sgc10\ndefNnkpUXtReqvkAcfnM07KnaQKBgQDMcrrg440Kc5X1UvqYuuNa8AKVQWgfaJye\nfLc6upuotY7uyaNToN2b7ZfMwcFJFoy6prZ5Xon/zv0xM97SqnLrz4EmlesUwpeI\nRFCC1/5+ophjujOSzNJ5rBT7/iGmuWTpYnJlFeBDeaGqVijFfZM3vPjtEpms1dk3\nhjDsqJxuDQKBgEjOo83cuvXYnTSuxiCiGCR5HbDiLR+/t8/HrknmK1AT09DfH+Ql\nF4tihez2zKuW/YlqDuOBdGM4r835M+NrWW4kyIpXJI306UEPHH2GwoQgT7A3NFXg\nnuUCmVWWOgGGMjskS/XGTfmv12AG10ayb7DxJ3JZzLPlLlve5nor6szpAoGBAImt\nNSWedusz3ScgYvsY12P0vniOXDTSeK3NpIIbChm5cfBhvufhK3sGq7PKQoQaeFh1\nTeo7fMjUNfK9UG8jM1KTIRC/4lfPlsW/40vJcmsKyX5W9MYFwjMHa/YqM7UXodn8\nXrat3aDytqiDbt9Xah0d16+mV/Bo7ecTKb0k6nq9AoGBAJmqDDkOPGOvh9s6PyfO\nHSETorXN0rmT807DpclmI876OUBwW8lNEpjgdPb7tTuGGP0UmTVFsrUEuD4YFWMc\nrE4Bc6QHKggBA69ao8DxKzEBEnrJmW6nUcRbxaLnMFWc4geS+YNMZSThhTDDvHRG\n29qomh1TmdASUoEocHKIeXm5\n-----END PRIVATE KEY-----\n",
    "client_email": "goog-sc-data-warehouse-392@nifty-might-404319.iam.gserviceaccount.com",
    "client_id": "114945514428373107829",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/goog-sc-data-warehouse-392%40nifty-might-404319.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
  }

# big query

SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive',
]

credentials = pydata_google_auth.get_user_credentials(
    SCOPES,
    auth_local_webserver=False,
)

df = pandas_gbq.read_gbq(
    "SELECT my_col FROM `my_dataset.my_table`",
    project_id='YOUR-PROJECT-ID',
    credentials=credentials,
)

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(
    '/path/to/nifty-might-404319-b6aac4c63637.json')