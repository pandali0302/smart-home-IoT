import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

df.head()
# deal with column time and convert it to datetime
df["time"] = pd.to_datetime(df["time"], unit="s")
df["time"].head()
df["time"] = pd.DatetimeIndex(
    pd.date_range("2016-01-01 05:00", periods=len(df), freq="30min")
)

# To utilize datetime information such as year, month and day in EDA and modeling phase, we need to extract them from time column.
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.weekday
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour
df["minute"] = df["time"].dt.minute

#  based on column hour, add a new column to indicate the different period (Night, Morning, Afternoon and Evening ) of a day.

#     Night : 22:00 - 23:59 / 00:00 - 03:59
#     Morning : 04:00 - 11:59
#     Afternoon : 12:00 - 16:59
#     Evening : 17:00 - 21:59

# We can create timing variable based on hour variable.
df["hour"].unique()
df["timing"] = "Night"
# df.loc[(df["hour"] >= 22) & (df["hour"] < 24), "timing"] = "Night"
df.loc[(df["hour"] >= 4) & (df["hour"] < 12), "timing"] = "Morning"
df.loc[(df["hour"] >= 12) & (df["hour"] < 17), "timing"] = "Afternoon"
df.loc[(df["hour"] >= 17) & (df["hour"] < 22), "timing"] = "Evening"
df.head()

df.info()
