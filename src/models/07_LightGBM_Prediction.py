# ----------------------------------------------------------------
# Modeling for Energy Consumption Prediction
# ----------------------------------------------------------------
"""
传统统计模型：适合处理具有季节性和趋势的时间序列数据 （数据标准化不是必需的）
    1. ARIMA    
        -- 适用于具有趋势和季节性的时间序列数据，结合了自回归、差分和移动平均。
    2. SARIMA   
        -- 适用于具有季节性的时间序列数据，是 ARIMA 的扩展。如月销售额、季节性温度变化
    3. VAR  
        --  适用于多变量时间序列数据，通过多个变量的过去值进行预测；适用于分析多个时间序列之间的动态关系和因果效应，常用于宏观经济数据的分析
    4. Prophet 
        -- Facebook 开发的一种时间序列预测模型，基于加性回归模型，可以捕捉时间序列的季节性、趋势和节假日效应。

机器学习模型：适合处理复杂特征和非线性关系，
    1. Linear Regression
    2. SVM  
        --  对数据的尺度敏感，特别是使用欧几里得距离作为核函数时，通常需要标准化
    3. LightGBM 
        -- 基于梯度提升的决策树算法，对特征的尺度不是非常敏感。不一定需要对数据进行归一化或标准化

深度学习模型：适合处理大规模数据和复杂时间序列模式
    1. LSTM -- RNN 变体，数据进行标准化或归一化，以加速收敛并提高模型性能
"""

# ----------------------------------------------------------------
# Load Libraries and dataset
# ----------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR

df = pd.read_pickle("../../data/interim/02_add_time_features.pkl")
df.info()

data = df.copy()


# ----------------------------------------------------------------
# Split data into training and testing sets
# ----------------------------------------------------------------
size = int(len(data) * 0.7)
train = data.iloc[:size]
test = data.iloc[size:]


# ----------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------
def evaluate_model(true, predicted):
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, predicted)
    mape = np.mean(np.abs((true - predicted) / true)) * 100
    r2 = r2_score(true, predicted)
    return mse, rmse, mae, mape, r2
