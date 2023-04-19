# SARIMA -> 比ARIMA 多了 seasonality
# Temperature Data is more suitable for SARIMA than ARIMA (with seasonality inside)
import statistics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from math import sqrt


# In general, if the data has clear seasonality, SARIMA may perform better,
# while if the data has a more gradual trend, triple exponential smoothing may be more appropriate.

#有缺失日期 並非連續就會出問題
df = pd.read_csv("Data/Temperature.csv", index_col='Date', parse_dates=True)
df.index = pd.to_datetime(df.index)
print(df.isna().sum())  # 找出個欄位的NA值數目
df['Min Temp'].fillna(method='bfill', inplace=True)
#取對數會幫助穩定 但是有負or0會出錯
print(df.head())
df = df['1985-01-01':'1987-12-31']

# 1. Check Stationarity
p_value = (adfuller(df['Min Temp'].dropna())[1])
if p_value < 0.05:
    print(f"p={p_value}\nData Set is Stationary")


# 2. Plot acf and pacf
# decide the p, q order for the above models (ACF-> MA; PACF->AR )
def acf_pacf():
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(df['Min Temp'].diff(), label=("Minimum Temperture in Malbourne"), color="blue")
    plot_acf(df['Min Temp'].diff().dropna(), lags=30)
    plot_pacf(df['Min Temp'].diff().dropna(), lags=30)
    plt.show()


# 3.Train Model
def train_SARIMA():
    frac = round(len(df.index) * 0.8)
    test = len(df.index) - frac
    df_train = df[:frac]
    df_test = df[frac:]
    print(df_train.tail())

    model_SARIMA = SARIMAX(df['Min Temp'], order=(1, 1, 1),seasonal_order=(1,0,1,12))
   #use predict() function
    prediction_SARIMA = model_SARIMA.fit().predict(start=len(df_train) + 1, end=len(df_train)+test, typ="levels")
    #start 必須指定train model的下一個 # typ="levels" 才會返回預測值而不是"difference"

    #以下兩種都可以:
    #forecast_SARIMA = model_SARIMA.fit().forecast(steps=24)
    forecast_SARIMA = model_SARIMA.fit().predict(start=len(df) + 1, end=len(df) + 24, typ="levels")

    print(model_SARIMA.fit().summary())
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(df_train['Min Temp'], label="Train")
    ax.plot(df_test['Min Temp'], label="Test", color="green")
    ax.plot(prediction_SARIMA, label="Prediction", color="orange")
    ax.plot(forecast_SARIMA, label="Forecast", color="red")
    nrmse = round(mean_squared_error(df_test['Min Temp'], prediction_SARIMA, squared=False)/statistics.mean(df_test['Min Temp']),2)
    mape=cal_mape(df_test['Min Temp'], prediction_SARIMA)
    ax.set_title(f" Minimum Temperature in Melbourne Prediction\nUsing SARIMA\nNRMSE={nrmse}  ;  MAPE = {mape}%")
    ax.set_ylim(-10, 30)
    ax.legend()
    plt.show()

def cal_mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = round(np.mean(np.abs((y_test - pred) / y_test))*100)
    return mape

acf_pacf()
train_SARIMA()


