---
subject: Use machine learning methods to predict stock market prices
primary-tools:
  - looker
  - dbt
  - airflow
  - python
  - tensorflow
  - big query
date-created: 2023-09-21
date-modified: 2023-10-04
cssclasses: 
aliases: 
status: active
tags:
  - project
  - portfolio
  - python
  - machine-learning
---

<center><h2>project progress: 15%</h2><progress value="15" max="100"></progress></center>

# People
	(Project-Manager:: Julia Williams)
	(Developer:: )
	(Team-Members:: Kevin Cruz, Kari Peterson)
	(Stakeholders:: )

# Objective
The objective of this project is to predict stock market prices with Machine Learning methods. 

# Problem Statement
The United States Stock Market faces fluctuations due to supply, demand, current events, etc. There are many variables that cause the stock market to experience random behavior. However, analyzing trends in the market from previous years and applying forecasting methods can allow systems with random behaviors to be predicted with high accuracy. In this case, the stock market is a system with random behavior. By using Long Short-Term Memory (LSTM) models (a Deep Learning network), and SARIMA (Seasonal Auto-Regressive Integrated Moving Average) models, stock prices can be predicted with high accuracy.

# Data

Twenty years of Yahoo Finance Data will be used. Data from Yahoo Finance will be the data source. 

Source: [Building an end-to-end data pipeline for stock forecasting using python](https://medium.com/@dana.fatadilla123/building-an-end-to-end-data-pipeline-for-stock-forecasting-using-python-63a857be11fe)

Github repo: https://github.com/readingshapes/stock_forecasting

# Method
LSTM (Long short-term memory) Models will be used to calculate future stock prices. LSTM Models are Recurrent Neural Networks (RNN) used for time-series data. 

Yahoo Finance Data -> Python -> SQL, BigQuery -> Airflow, dbt -> TensorFlow -> Looker

# Tools
- LANGUAGE: Python
- DATABASE: SQL
- DATA WAREHOUSE: Google BigQuery
- ML MODELING: Google Looker
- SOFTWARE LIBRARY FOR ML: TensorFlow 
- PLATFORM FOR SCHEDULING BATCHES: Apache Airflow 
- DATA BUILD TOOL: dbt

# Method
LSTM (Long short-term memory) Models will be used to calculate future stock prices. LSTM Models are Recurrent Neural Networks (RNN) used for time-series data. 

Yahoo Finance Data -> Python -> SQL, BigQuery -> Airflow, dbt -> TensorFlow -> Looker

# Final Product Description
TBD

# Research
TBD

# Files
TBD

# Tasks
TBD

# Status Updates
12-4
1. Julia is thinking of having the program create a report at the end that displays the top 50 stocks for January 2024. Julia wants to first design the program to work for one stock, then will reapproach for multiple stocks? 
3. Kevin needs to download vscode, and all necessary libraries needed to run code
4. Goal this week: Julia and Kevin will research more about LTSM models


