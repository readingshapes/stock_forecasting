
'''
title:
autho:
date:
description:
    - 20 years of data
    - lstm models
'''


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
#%pip install statsmodels
#%pip install tensorflow
#%pip install yfinance
#%pip install stockstats
import yfinance as yf
from stockstats import StockDataFrame as sdf

# step 1: import data

stock_data = yf.download("GOOGL" , start = "2018-01-01" , interval = '1d')
print(stock_data.head())

stock_df = sdf.retype(stock_data)
stock_data[['stochrsi', 'macd', 'mfi']]=stock_df[['stochrsi', 'macd', 'mfi']]

print(stock_data)


# step 2:



