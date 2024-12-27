# Stock Forecasting with SARIMA Model

A project by Julia Williams and Kevin Cruz

## Inspiration
Portfolio project taken from: [Building an End-to-End Data Pipeline for Stock Forecasting using Python](https://medium.com/@dana.fatadilla123/building-an-end-to-end-data-pipeline-for-stock-forecasting-using-python-63a857be11fe)

## Objective
The objective of this project is to predict stock market prices using the **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) model. The goal is to evaluate the performance of this classical time-series forecasting model on stock price prediction for three different companies: **Apple (AAPL)**, **Tesla (TSLA)**, and **Nvidia (NVDA)**. By comparing SARIMA's predictive accuracy across different time ranges, we aim to understand its strengths and limitations for financial forecasting.

This project uses **SARIMA** models, a statistical time-series method, to capture the seasonal and cyclical behavior inherent in stock price movements. The performance of the model is evaluated against actual historical stock data, with emphasis on accuracy, robustness, and interpretability.

## Data
Yahoo Finance API was used to collect historical stock data for the project. The stock data for three companies—**Apple (AAPL)**, **Tesla (TSLA)**, and **Nvidia (NVDA)**—was downloaded and analyzed for training and forecasting.

## Machine Learning Model: SARIMA
### SARIMA (Seasonal AutoRegressive Integrated Moving Average)
The **SARIMA model** is a classical time-series forecasting technique that builds on the ARIMA model by incorporating seasonal components. This model is particularly useful for time series data with strong seasonal patterns, such as stock market prices. SARIMA consists of several parameters that define the seasonal autoregressive (AR), differencing (I), and moving average (MA) components, as well as seasonal components that account for periodic patterns in data.

For each stock in this project, SARIMA models were built with the following objectives:
1. **Forecast stock prices** over a defined future period.
2. **Evaluate model accuracy** using various time ranges and key metrics.
3. **Identify the best model parameters** for different stocks (AAPL, TSLA, NVDA).

## Methodology
### Time Range Evaluation
The SARIMA models were evaluated using four different time ranges for each stock:
- **01-01-2015 to 05-31-2024**
- **01-01-2017 to 05-31-2024**
- **01-01-2019 to 05-31-2024**
- **01-01-2021 to 05-31-2024**

Each model’s performance was analyzed in terms of forecast accuracy. The most effective time range was selected for each stock based on the evaluation metrics (RMSE, MSE, MAPE, etc.).

For example:
- **Apple (AAPL)**: Best accuracy was achieved with the **2019** time range.
- **Tesla (TSLA)**: Best accuracy was achieved with the **2021** time range.
- **Nvidia (NVDA)**: Best accuracy was achieved with the **2019** time range.

### Accuracy Metrics
The SARIMA models were evaluated using several performance metrics:
- **RMSE** (Root Mean Squared Error)
- **MSE** (Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **AIC** (Akaike Information Criterion)
- **BIC** (Bayesian Information Criterion)

These metrics were used to assess the models' predictive accuracy by comparing the forecasted stock prices against the actual values.

### Data Preprocessing
Before applying the SARIMA models, the data was preprocessed:
- Stock price data was retrieved using Yahoo Finance.
- Missing values were handled, and weekend data (non-business days) was removed.
- The stock price data was split into training and testing datasets for evaluation purposes.

### Train-Test Split Evaluation for SARIMA
To ensure robust evaluation, the dataset was divided into **rolling train-test splits**, where the training data size progressively increases over time. This ensures that the models are trained on the most up-to-date data and evaluated on future unseen data, preventing data leakage.

For each rolling split, the model was trained on a subset of the data and tested against the corresponding future period. The effectiveness of the model was evaluated for each period based on its ability to predict stock prices accurately.

## Project Infrastructure
Yahoo Finance Data -> Python -> SQL, BigQuery -> Airflow, dbt -> Looker

## Tools
- **LANGUAGE**: Python
- **DATA WAREHOUSE**: Google BigQuery
- **SOFTWARE LIBRARY FOR ML**: **pmdarima**, **statsmodels**
- **PLATFORM FOR SCHEDULING BATCHES**: Apache Airflow
- **DATA BUILD TOOL**: dbt

## Step-by-Step Guide to Implementing the SARIMA Model

### Step 1: Install Required Packages

To start, make sure to install the necessary Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib pmdarima yfinance statsmodels
