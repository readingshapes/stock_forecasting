##STOCK FORECASTING WITH SARIMA MODELS

Julia Williams, Kevin Cruz

# stock_forecasting
Portfolio project taken from: [Building an End-to-End Data Pipeline for Stock Forecasting using Python](https://medium.com/@dana.fatadilla123/building-an-end-to-end-data-pipeline-for-stock-forecasting-using-python-63a857be11fe)

# Objective

The objective of this project is to predict stock market prices with Machine Learning methods. 

# Data

Yahoo Finance package was used for historical data. 

# Method
SARIMA (Seasonal AutoRegressive Integrated Moving Average) Models were created for three stocks: AAPL, NVDA and TSLA. SARIMA models with four different time ranges of historical data were compared for each stock. The accuracy of these models were analyzed. 

A LSTM (Long short-term memory) Model was created for the AAPL stock for more analysis. LTSM Models are Recurrent Neural Networks (RNN) used for time-series data.

Project Infrastructure:
Yahoo Finance Data -> Python -> SQL, BigQuery -> Airflow, dbt -> TensorFlow -> Looker

# Tools
- LANGUAGE: Python
- DATA WAREHOUSE: Google BigQuery
- ML MODELING: Google Looker
- SOFTWARE LIBRARY FOR ML: TensorFlow 
- PLATFORM FOR SCHEDULING BATCHES: Apache Airflow 
- DATA BUILD TOOL: dbt
