# stock_forecasting
Portfolio project taken from: [Building an End-to-End Data Pipeline for Stock Forecasting using Python](https://medium.com/@dana.fatadilla123/building-an-end-to-end-data-pipeline-for-stock-forecasting-using-python-63a857be11fe)

# Objective

The objective of this project is to predict stock market prices with Machine Learning methods. 

# Data

Twenty years of Yahoo Finance Data will be used. Data from Yahoo Finance will be the data source. 

# Method
LSTM (Long short-term memory) Models will be used to calculate future stock prices. LTSM Models are Recurrent Neural Networks (RNN) used for time-series data. 

Yahoo Finance Data -> Python -> SQL, BigQuery -> Airflow, dbt -> TensorFlow -> Looker

# Tools
- LANGUAGE: Python
- DATABASE: PostgreSQL or SQL?
- DATA WAREHOUSE: Google BigQuery
- ML MODELING: Google Looker
- SOFTWARE LIBRARY FOR ML: TensorFlow 
- PLATFORM FOR SCHEDULING BATCHES: Apache Airflow 
- DATA BUILD TOOL: dbt