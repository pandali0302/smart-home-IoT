"""
深度学习模型可以解决基本上所有时序问题，而且模型可以自动学习特征工程，极大减少了人工；不过需要较高的模型架构能力。

深度学习类模型主要有以下特点：

    不能包括缺失值，必须要填充缺失值，否则会报错；
    支持特征交叉，如二阶交叉，高阶交叉等；
    需要 embedding 层处理 category 变量，可以直接学习到离散特征的语义变量，并表征其相对关系；
    数据量小的时候，模型效果不如树方法；但是数据量巨大的时候，神经网络会有更好的表现；

LSTM 模型是一种特殊的 RNN 模型，它可以学习长期依赖关系，适合处理时间序列数据。LSTM 模型的核心是单元状态和隐藏状态，通过这两个状态，模型可以学习到时间序列数据的长期依赖关系。

LSTM 模型的输入是一个三维张量，形状为 (samples, time_steps, features)，其中：
    
        samples：样本数
        time_steps：时间步数
        features：特征数

LSTM 模型的输出是一个二维张量，形状为 (samples, features)，其中：

        samples：样本数
        features：特征数


# LSTMs for Multivariate Time Series Forecasting
多变量模型使用目标变量和天气特征的历史值来进行预测。这种模型可以更好地捕捉目标变量和特征之间的关系，从而提高预测的准确性。

"""

# ----------------------------------------------------------------
# import packages and modules
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------
# load the dataset
# ----------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

df.info()

data = df.copy()
# 重采样为每日数据
daily_data = data.resample("D").mean()


# ----------------------------------------------------------------
# Split data into training and testing sets
# ----------------------------------------------------------------
train_size = int(len(daily_data) * 0.8)
train_data = daily_data[:train_size]
test_data = daily_data[train_size:]

# ----------------------------------------------------------------
# feature scaling
# ----------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

weather_features = [
    "temperature",
    "humidity",
    "visibility",
    "pressure",
    "windSpeed",
    "cloudCover",
    "windBearing",
    "precipIntensity",
    "dewPoint",
    "precipProbability",
]

target = ["use_HO"]


# 选择特征列和目标列
features = weather_features + target

# 分别对训练集和测试集进行缩放
scaler = MinMaxScaler()

# 仅使用训练集拟合缩放器
scaler.fit(train_data[features])

# 对训练集和测试集进行缩放
scaled_train_data = scaler.transform(train_data[features])
scaled_test_data = scaler.transform(test_data[features])

scaled_train_df = pd.DataFrame(
    scaled_train_data, columns=features, index=train_data.index
)
scaled_test_df = pd.DataFrame(scaled_test_data, columns=features, index=test_data.index)


# ----------------------------------------------------------------
# 创建序列
# Transform the time series into a supervised learning problem
# ----------------------------------------------------------------
"""
通过创建序列，可以为模型提供过去一段时间的数据，使其能够更好地捕捉到这种依赖性。
时间序列数据通常是连续的，为了使模型能够学习，必须将连续的时间点组织成输入-输出对。例如，假设我们使用过去60分钟(序列长度)的数据来预测下一分钟的值，数据将被组织成如下形式：

    输入：过去60分钟的数据（特征值） t1, t2, t3, ..., t60
    输出：下一分钟的目标值          t61

这种结构的输入和输出对可以用于训练监督学习模型。

- 如果数据有明显的周期性，例如每周或每月的周期，选择一个能覆盖一个或多个周期的序列长度会很有帮助。例如，电力消耗数据可能具有周周期，使用7天作为序列长度可能会捕捉到一周内的模式。
- 较长的序列长度意味着更多的输入特征，这可能增加模型的复杂性和计算需求。如果数据量较大，过长的序列可能会导致训练时间过长或者过拟合问题。
- 如果数据集较小，使用较短的序列长度可能更实际，以确保有足够的训练样本。

"""
import numpy as np


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)


seq_length = 7  # 选择一周的序列长度

# 创建训练集和测试集的序列
train_sequences = create_sequences(scaled_train_df.values, seq_length)
test_sequences = create_sequences(scaled_test_df.values, seq_length)

# 拆分输入和输出
X_train, y_train = (
    train_sequences[:, :-1],
    train_sequences[:, -1][:, -1],
)  # y使用最后一个时间步的目标值use_HO
X_test, y_test = test_sequences[:, :-1], test_sequences[:, -1][:, -1]

# 调整输入形状以适应LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
