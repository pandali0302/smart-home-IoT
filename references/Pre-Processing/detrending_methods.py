"""
Choosing the appropriate detrending method depends on the characteristics of the data and the application needs:

- **Moving Average**: Simple and easy to use, suitable for short-term smoothing.
- **Differencing**: Suitable for removing linear trends but may introduce additional noise.
- **Seasonal Decomposition**: Suitable for time series with clear seasonality and periodicity.
- **Polynomial Fitting**: Suitable for capturing nonlinear trends.
- **Rolling Window Method**: Suitable for removing local trends, best for relatively stable data.

In practice, it is often beneficial to try and compare multiple methods to determine the most effective approach for detrending.

移动平均: 简单易用，适用于短期平滑。适用于平稳数据，但对突变和非线性趋势处理较差
差分法: 适用于去除线性趋势，但可能引入额外的噪音。适用于线性趋势数据，但高次差分容易引入噪声
多项式拟合: 适用于捕捉非线性趋势。但阶数选择不当容易过拟合
季节性分解: 适用于具有明显季节性和周期性的时间序列。

在方法上，去趋势通常使用移动平均法、差分法和多项式拟合法等；而季节性调整则通常使用季节性分解方法（如 STL 分解）和回归模型等

如果数据中存在明显的季节性或周期性波动，直接去趋势可能导致错误的分析结果。应该先进行季节性调整，再进行去趋势处理

"""

# ----------------------------------------------------------------
# 1. Moving Average 移动平均
# ----------------------------------------------------------------
"""
Moving average is a simple and commonly used detrending method. It smooths the data and removes the trend by calculating the average of neighboring points in the time series.
"""
import pandas as pd

# Load the data
data = pd.read_csv("energy_weather_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data.set_index("timestamp", inplace=True)

# Calculate moving average (window=7 represents a weekly window)
data["moving_avg"] = data["house_energy"].rolling(window=7).mean()

# Detrend (subtract the moving average)
data["detrended"] = data["house_energy"] - data["moving_avg"]


# ----------------------------------------------------------------
# 2. Differencing  差分法
# ----------------------------------------------------------------
"""Differencing removes the trend by calculating the difference between consecutive values in the time series."""

# Calculate the first difference
data["differenced"] = data["house_energy"].diff()

# Remove missing values
data["differenced"].dropna(inplace=True)

# ----------------------------------------------------------------
# 3. Seasonal Decomposition 季节性分解
# ----------------------------------------------------------------
"""Seasonal decomposition separates the time series into trend, seasonal, and residual components, thereby removing the trend."""

from statsmodels.tsa.seasonal import seasonal_decompose

# Seasonal decomposition (period=365 represents an annual cycle)
result = seasonal_decompose(data["house_energy"], model="additive", period=365)
data["trend"] = result.trend
data["seasonal"] = result.seasonal
data["residual"] = result.resid

# Detrend (using the residual component)
data["detrended"] = data["house_energy"] - data["trend"]

# ----------------------------------------------------------------
#  4. Polynomial Fitting 多项式拟合
# ----------------------------------------------------------------
"""Polynomial fitting captures the trend by fitting a polynomial function to the data and then removing the fitted values from the time series."""

import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Fit a polynomial (degree=2 for a quadratic polynomial)
x = np.arange(len(data))
y = data["house_energy"].values
p = Polynomial.fit(x, y, 2)

# Calculate the fitted values
data["poly_trend"] = p(x)

# Detrend (subtract the fitted values)
data["detrended"] = data["house_energy"] - data["poly_trend"]
