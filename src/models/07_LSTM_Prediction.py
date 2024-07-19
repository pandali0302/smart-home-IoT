"""
深度学习模型可以解决基本上所有时序问题，而且模型可以自动学习特征工程，极大减少了人工；不过需要较高的模型架构能力。

深度学习类模型主要有以下特点：

    不能包括缺失值，必须要填充缺失值，否则会报错；
    支持特征交叉，如二阶交叉，高阶交叉等；
    需要 embedding 层处理 category 变量，可以直接学习到离散特征的语义变量，并表征其相对关系；
    数据量小的时候，模型效果不如树方法；但是数据量巨大的时候，神经网络会有更好的表现；

LSTM 模型是一种特殊的 RNN 模型，它可以学习长期依赖关系，适合处理时间序列数据。LSTM 模型的核心是单元状态和隐藏状态，通过这两个状态，模型可以学习到时间序列数据的长期依赖关系。

divide the sequence into multiple inputs/output patterns, converting the sequence to a supervised learning problem. From the division of data in the input/output pattern, the model will learn about the input patterns and the output.

"""

# ----------------------------------------------------------------
# import packages and modules
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ----------------------------------------------------------------
# load the dataset
# ----------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df.info()

data = df.copy()
