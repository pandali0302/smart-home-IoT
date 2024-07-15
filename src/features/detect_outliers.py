import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import seasonal_decompose
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_add_time_features.pkl")
df.info()

outlier_columns = df.columns[:13]


# ----------------------------------------------------------------
# function to return plots for the feature
# ----------------------------------------------------------------
cp = df.copy()
outlier_columns = cp.columns[:13]


def normality(data, feature):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.kdeplot(data[feature])
    plt.subplot(1, 2, 2)
    stats.probplot(data[feature], plot=pylab)
    plt.show()


for col in outlier_columns:
    normality(cp, col)

normality(cp, "use_HO")
cp["use_HO_log"] = np.log(df["use_HO"])
normality(cp, "use_HO_log")


# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------
# Insert IQR function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plot a single column
col = "use_HO"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# ----------------------------------------------------------------
# Z-score (distribution based)
# ----------------------------------------------------------------
def mark_outliers_zscore(dataset, col):
    """Function to mark values as outliers using the Z-score method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    z_scores = np.abs(stats.zscore(dataset[col]))
    threshold = 3

    dataset[col + "_outlier"] = z_scores > threshold

    return dataset


# Plot a single column
col = "use_HO"
dataset = mark_outliers_zscore(df, col)
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_zscore(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------
# Insert LOF function
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


col = "use_HO"
dataset, outliers, X_scores = mark_outliers_lof(df, [col])
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
)

# Loop over all columns
dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

df.info()
# ----------------------------------------------------------------
# Machine Learning Models:
# Using models to predict normal consumption and flag anomalies
# (e.g., a sudden increase in fridge consumption might indicate a malfunction).
# ----------------------------------------------------------------
# use machine learning models to predict normal consumption and flag anomalies.
# use anomaly detection algorithms such as Isolation Forest.
from sklearn.ensemble import IsolationForest


def mark_outliers_isolation_forest(dataset, columns, contamination=0.01):
    """Mark values as outliers using Isolation Forest

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        contamination (float, optional): The proportion of outliers in the data.
        Defaults to 0.01.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    data = dataset[columns]
    outliers = iso_forest.fit_predict(data)
    # -1 indicates an anomaly, 1 indicates normal
    dataset["outlier_iso_forest"] = outliers == -1
    return dataset, outliers


# Plot a single column
col = "use_HO"
dataset, _ = mark_outliers_isolation_forest(df, [col])
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col="outlier_iso_forest", reset_index=True
)


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

time_series_plot(df[["use_HO"]])

# ----------------------------------------------------------------
# Detrending and Deseasonalizing
# ----------------------------------------------------------------
"""
Removing trends helps isolate sudden changes by eliminating long-term variations that could mask these anomalies. Methods like differencing and seasonal decomposition are effective:

    Differencing: Removes linear trends and helps highlight abrupt changes.

    Seasonal Decomposition: Separates the time series into trend, seasonal, and residual components, allowing you to focus on the residual (which contains the anomalies).
"""
# ----------------------------------------------------------------
# Seasonal Decomposition: Suitable for time series with clear seasonality and periodicity.
# ----------------------------------------------------------------
# copy the data
df_sd = df.copy()
df_sd = df_sd[["use_HO"]].resample("D").mean()

# Seasonal decomposition
result = seasonal_decompose(df_sd["use_HO"], model="additive")
df_sd["trend"] = result.trend
df_sd["seasonal"] = result.seasonal
df_sd["residual"] = result.resid  #  using the residual component
# df_sd["detrended"] = df_sd["use_HO"] - df_sd["trend"]

# plot to visualize the components
plt.figure(figsize=(15, 5), dpi=100)
sns.lineplot(x=df_sd.index, y=df_sd["residual"])
# sns.lineplot(x=df_sd.index, y=df_sd["trend"])
# sns.lineplot(x=df_sd.index, y=df_sd["seasonal"])
plt.title("Seasonal Decomposition")
plt.legend(["Residual", "Trend", "Seasonal"])
plt.show()

# Visualization per component
plt.figure(figsize=(10, 6))
plt.subplot(411)
plt.plot(df_sd["use_HO"], label="Original")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(df_sd["trend"], label="Trend")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(df_sd["seasonal"], label="Seasonal")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(df_sd["residual"], label="Residual")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# 处理 NaN 值 - 方法示例
df_sd.isnull().sum()
# 选择其中一种方法进行处理
# 删除 NaN 值
# data_cleaned = data.dropna()
# 前向填充
df_sd["residual"].fillna(method="ffill", inplace=True)
# 后向填充
df_sd["residual"].fillna(method="bfill", inplace=True)
# 插值
df_sd["residual"] = df_sd["residual"].interpolate(method="linear")
# 填充为零
# df_sd['residual'].fillna(0, inplace=True)
# 填充为均值
# df_sd['residual'].fillna(df_sd['residual'].mean(), inplace=True)

# ----------------------------------------------------------------
# Differencing: Removes linear trends and helps highlight abrupt changes.
# ----------------------------------------------------------------

df_diff = df.copy()
df_diff = df_diff[["use_HO"]].resample("D").mean()

df_diff["differenced"] = df_diff["use_HO"].diff().dropna()

# plot to visualize the differenced time series
plt.figure(figsize=(15, 5), dpi=100)
sns.lineplot(x=df_diff.index, y=df_diff["differenced"])
plt.title("Differenced Time Series")
plt.show()


# ----------------------------------------------------------------
# Denoising
# ----------------------------------------------------------------
"""
Reducing noise helps make anomalies more evident. Moving averages and low-pass filters can smooth out short-term fluctuations while preserving significant changes.

    Moving Average: Smooths the data, reducing noise but potentially also smoothing out minor anomalies.

    Low-pass Filter: More sophisticated than moving average, can preserve more of the original signal's characteristics.
"""

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()

df_lowpass = df_lowpass[app_columns].resample("D").mean()
col = "use_HO"
df_lowpass = df_lowpass[[col]]
LowPass = LowPassFilter()

fs = 1  # Sampling frequency
cutoff = 0.25  # higher cutoff frequency, more close to raw data

df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)

# Visualization -1 : compare raw data and filtered data in different plot
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(df_lowpass[col].reset_index(drop=True), label="raw data")
ax[1].plot(
    df_lowpass[col + "_lowpass"].reset_index(drop=True), label="butterworth filter"
)
ax[0].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    fancybox=True,
    shadow=True,
)
ax[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    fancybox=True,
    shadow=True,
)

# Visualization - 2 : compare raw data and filtered data in the same plot
plt.figure(figsize=(15, 5))
plt.plot(df_lowpass[col], label="Original")
plt.plot(df_lowpass[col + "_lowpass"], label="Filtered", color="red")
plt.legend()
plt.show()

df_lowpass[col] = df_lowpass[col + "_lowpass"]
del df_lowpass[col + "_lowpass"]

# for col in app_columns:
#     df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
#     df_lowpass[col] = df_lowpass[col + "_lowpass"]
#     del df_lowpass[col + "_lowpass"]

# Plot a single column
col = "use_HO"
dataset, _ = mark_outliers_isolation_forest(df_lowpass, [col])
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col="outlier_iso_forest", reset_index=True
)


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col = "use_HO"
dataset, _ = mark_outliers_isolation_forest(df_lowpass, [col])
dataset.loc[dataset["outlier_iso_forest"], col] = np.nan


# Create a loop

outlier_removed_df = df.copy()

for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_isolation_forest(df[df["label"] == label], col)

        # Replace values marked as outliers with Nan
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # update the column in the original dataframe
        outlier_removed_df.loc[(outlier_removed_df["label"] == label), col] = dataset[
            col
        ]

        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")

outlier_removed_df.info()


# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outlier_removed_df.to_pickle("../../data/interim/02-outlier_removed_chauvenet.pkl")
