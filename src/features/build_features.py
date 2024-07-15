import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


# ----------------------------------------------------------------
# Time-Series Analysis / Seasonality and Trend
# ----------------------------------------------------------------
# Iterate each appliance, plot the time series, resample to daily
def time_series_plot(data):
    data_resampled = data.resample(
        "D"
    ).mean()  # Resample time to day and calculate the mean
    for i in data_resampled.columns:
        plt.figure(figsize=(15, 5), dpi=100)
        sns.lineplot(x=data_resampled.index, y=data_resampled[i], label=i)
        plt.title(f"{i} Time Series")
        plt.legend()
        plt.show()


app_columns = df.columns[:13]
weather_columns = df.columns[13:-7]
time_series_plot(df[app_columns])
#  some appliances have seasonality, some have trend, some have both
#  Furnace has a trend, Wine cellar has seasonality, Fridge has both

time_series_plot(df[weather_columns])
#  temperature and dewPoint have seasonality, visibility has trend
