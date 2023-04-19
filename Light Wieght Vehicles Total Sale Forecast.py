import statistics
import pandas as pd
import numpy as np
from matplotlib import rcParams
from cycler import cycler
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#Triple Exponential Smoothing -> 成本最低最有效的 model for Data with Trend and Seasonal

df = pd.read_csv('Data/LTOTALNSA.csv', index_col='DATE', parse_dates=True)
df.index.freq = 'MS'
df = df['2010-01-01':'2020-01-01']
rcParams['figure.figsize'] = 18, 7
rcParams['axes.spines.top'] = False  # 上方骨架無邊框
rcParams['axes.spines.right'] = True  # 右邊骨架有邊框
rcParams['axes.prop_cycle'] = cycler(color=['#01814A'])  # 線條顏色
rcParams['lines.linewidth'] = 1  # 線條粗度
rcParams['font.size'] = '10'

def train_ltotalnsa():
#劃分訓練集和測試集
#for 隨機劃分 : df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
#Time Series 必須用前後劃分
    frac = round(len(df.index)*0.8)
    test = len(df.index)-frac
    df_train = df[:frac]
    df_test = df[frac:]

    model_mul_mul = ExponentialSmoothing(df_train['LTOTALNSA'], trend='mul', seasonal='mul', seasonal_periods=12)
    predictions_mul_mul = model_mul_mul.fit().forecast(steps=test)

    #預測結果和測試集的殘差值
    nrmse_mul_mul = round(mean_squared_error(df_test['LTOTALNSA'], predictions_mul_mul, squared=False)/statistics.mean(df_test['LTOTALNSA']),2)
    mape = cal_mape(df_test['LTOTALNSA'], predictions_mul_mul)
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(df_train['LTOTALNSA'], label="Training Data")
    ax.plot(df_test['LTOTALNSA'], label="Testing Data",color="gray")
    ax.plot(predictions_mul_mul, label="Prediction",color="orange")
    ax.legend()
    ax.set_title(f"Light Weight Vehicle Sales Prediction \nUsing Triple Exponential Smoothing \nNRMSE={nrmse_mul_mul} ;  MAPE = {mape}%" )

    #用整個df當作訓練set 預測更未來
    model_mul_mul = ExponentialSmoothing(df['LTOTALNSA'], trend='mul', seasonal='mul', seasonal_periods=12)
    predictions_future = model_mul_mul.fit().forecast(steps=24) #預測未來一年
    ax.plot(predictions_future, label="Forecast",color="red")
    ax.minorticks_on() #小刻度線
    ax.legend()
    plt.show()

def cal_mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = round(np.mean(np.abs((y_test - pred) / y_test))*100)
    return mape

train_ltotalnsa()