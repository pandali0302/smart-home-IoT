import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_add_time_features.pkl")
df.info()
df.head()
df.describe()


"""    

选择相关特征: 根据业务需求和数据的相关性选择重要特征，减少噪音和冗余数据。
    特征交互: 生成新的特征，如天气数据的交互特征，来捕捉更多信息。


# 示例：创建温度与湿度的交互特征
data['temp_humidity_interaction'] = data['temperature'] * data['humidity']

"""
