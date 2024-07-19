
# ----------------------------------------------------------------
# 数据预处理:
    - 数据归一化
    数据归一化是预处理中的重要环节，特别是对于 LSTM 这种对输入数据敏感的模型。我们通常使用 MinMaxScaler 来将数据缩放到一个指定的范围，通常是 0 到 1。这样可以帮助模型更快地收敛，同时避免梯度消失或爆炸的问题。
# ----------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
import numpy as np


scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ----------------------------------------------------------------
# 时间序列特征工程

对于时间序列数据，我们可能还需要执行一些特征工程，比如创建滞后特征（即基于之前的时间点的数据），这在模型需要捕捉时间序列的自相关性时非常有用。
# ----------------------------------------------------------------




# ----------------------------------------------------------------
# LSTM Data Preparation
# ----------------------------------------------------------------

# 1.Transform the time series into a supervised learning problem
# 2.Transform the time series data so that it is stationary.
# 3.Transform the observations to have a specific scale.





# ----------------------------------------------------------------
# LSTMs for Univariate Time Series Forecasting
# ----------------------------------------------------------------


