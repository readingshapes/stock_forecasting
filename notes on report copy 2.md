# notes on report copy 2

*What is the question?*
- *Is the question: "Should we add the SARIMA model to our stock forecasting toolbox?"*
    - *If so, the answer is yes or no. The first sentence in the summary should be either:*

*"You should add the SARIMA model to your toolbox because..."* 
OR
*"The SARIMA model is not a good fit for our stock forecasting process because..."*

## summary
The Seasonal Autoregressive Integrated Moving Average (SARIMA) model provides a robust framework for predicting stock prices. *and is a good complement to the existing machine learning models used by x.*

### Definition 
**SARIMA Model:** Seasonal Autoregressive Integrated Moving Average (SARIMA) model, a time series forecasting method, is implemented to capture the underlying patterns and trends in the stock price data.

### Benefits
SARIMA's ability to capture the seasonal patterns and cyclic behavior inherent in sequential data, combined with its foundation in statistical time series analysis principles, enhances the credibility of ~~our predictions.~~ *stock forecasting processes.*

The model's ~~interpretable parameters~~ *use simpler words* provide a deeper understanding of the underlying dynamics driving stock price movements. *how?*

*list top three reasons to use this model*

### Caveats
However, it is important to note SARIMA's limitation in capturing external factors, such as news events or sudden market shocks, which may affect stock prices but are not directly incorporated into the model. Despite this limitation, SARIMA remains a valuable tool for predicting stock prices based on historical patterns and internal data dynamics.

*why do people not already use it?*

## methodology - more detail: specifics about this demo
For this project, four time ranges of historical data were evaluated to ~~assess the accuracy of the SARIMA models' predictions~~ *to determine the optimal time lag and time steps to include in the model* for three different stocks (AAPL, TSLA, and NVDA). *what companies are these stocks for? why did you pick them? what makes them good for testing? Do they cover the range of volatility and/or new/old stocks? What are the key indicators of stocks?*

Predicted Dates: 06-01-2024 to 06-11-2024

Historical Data Time Ranges:

1. 01-01-2015 to 5-31-2024
2. 01-01-2017 to 5-31-2024
3. 01-01-2029 to 5-31-2024
4. 01-01-2021 to 5-31-2024

### answer: which time range works best
*which one, 2019 or 2021? accuracy threshold % for each stock?*
*for whichever year you pick, list accuracy % for each stock*

### **variance among stocks?**

- They demonstrated high accuracy in capturing seasonal trends and underlying patterns in stock data. *see above*
~~- They provided interpretable parameters that offered valuable insights into stock price movements.~~ *what does this really mean?*

~~Other time ranges (2015, 2017) were less effective because:~~
~~- They showed slower computation times.~~
~~- They had lower accuracy metrics compared to recent time ranges.~~
~~- They required more computational resources, making them less practical for real-time stock prediction.~~ 
*this is not necessary. it's implicit that these don't meet criteria. when i suggested you include something about models not selected, that doesn't mean date ranges which are about finetuning a single model*

### Tests used to determine accuracy
Prediction *accuracy* was determined using comparison metrics, *including* ~~such as~~ R-squared, p-value, RMSE, MSE, MAPE, AIC, and BIC.

### Factors for Selecting Time Ranges:
1. Accuracy (RMSE, MSE, MAPE) *which? why?*
    1a. *seasonality*
2. Speed
3. Compute resource usage considerations
4. Reproducibility
5. World Events
6. Other factors *?*

Time Series Prediction vs. Other Machine Learning Tasks: 
~~SARIMA excels in time series analysis, making it highly effective for stock price predictions, where the temporal dependencies of historical data play a crucial role. In contrast, other machine learning models may struggle to capture these temporal patterns as effectively.~~
*sarima is one of many options for time series. what **specifically** makes it useful?*

### Specifics of the Model:
- SARIMA: Captures seasonality and ~~trends in data~~, *trends in data aside from seasonality? like what?*

 making it robust for stocks with clear cyclical patterns, such as AAPL and TSLA. *because why? what are these stocks? what are their names?*
 
 ~~By leveraging its interpretable parameters, SARIMA provides a nuanced understanding of stock price movements and helps identify recurring trends.~~ *This is the third time you said this, and i still don't understand what it means*

- Contextual Effectiveness:
SARIMA's ability to effectively utilize recent historical data has proven beneficial for stable stocks like AAPL. *how? why is APPL stable?*

However, for more volatile stocks like NVDA and TSLA, 
*how are these volatile?*

the model's performance was enhanced by incorporating longer historical data, which improved prediction accuracy by accounting for more complex stock price fluctuations. *what longer historical data? does that mean a different model would work better for volatile stocks?*

### Definition Metric choices:
*we considered using the following metrics to determine accuracy:*
- R-squared - definition
- p-value - definition
- RMSE - definition
- MSE - definition
- MAPE - definition
- AIC - definition
- BIC - definition

### Selected definition of accuracy: 
The evaluation metrics employed include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), *why?*

and prediction accuracy. *what does this mean?*

In this context, accuracy refers to how closely the SARIMA model's predicted values match the actual observed stock prices, as measured by low MSE, RMSE, and MAPE scores. *what are these? what is the definition of low?*

### fit assessment
*where is this?*

## Step-by-Step Guide to Implementing the SARIMA Model
### Step 1: Necessary packages were imported into Python. ~~Packages important to note were pandas, numpy, yfinance, etc.~~

~~# Step 1: Import packages~~ *this is duplicated*
~~# % pip install pandas~~
~~# % pip install numpy~~
~~# % pip install plt~~
*you don't need to show pip install in documentation*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
*you need to import scikit-learn before you import modules*

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pmdarima as pm
from sklearn.metrics import mean_squared_error *this is imported twice*
import numpy as np *this is imported twice*
import statsmodels.api as sm

*the package list should be in order from most common to least, with modules listed last*

### Step 2: User input for stock ticker
*why is there input here? how are you presenting this? as a black box with ux? if not, skip this and define your variables at the top of the file*

### Step 3: Check data before processing
*how did you check this? what are the indicators?*

~~Step 4: The index of the stock_data DataFrame was first converted to a DateTimeIndex to enable easy filtering. Then, weekend data was filtered out by retaining only rows where the day of the week was less than 5 (Monday to Friday).~~ *not necessary to list this as a step*

The function took stock_transformed as input and processed its "Close" price data.
*this would be better defined as: the function uses the close price of the stock dataframe*

### Step 5: Define SARIMA model
#### Parameters:
- SARIMA: (s)easonal + (i)ntegrated + (a)uto (r)egression + (m)oving (a)verage
    - p = number of lagged values included:
    - q = number of time steps in moving average:
    - d = number of differencing sequences: 
    - m = number of observations per year:
    - smoothing choice: explain
    - final model choice: SARIMA(p, d, q)(P, D, Q)m
- p: 1 - 3
- q: 1 - 3
- d: none
- m: 12
- P: 0
- D: 1
- Q: ?

#### Fit the SARIMA model
*how? what does this mean?*

### Step 6: Run predictions
- n_forecast: 
- forecast:
- conf_int:
*what are the variables? why did you make them?*

### Step 7: The output was assessed.
*how? what does this mean? what are the findings?*

### Step 8: Chart actual against forecast

### Step 9: Assess selected metrics
- RMSE
- MSE
- MAPE

*show values and explain what they mean*









