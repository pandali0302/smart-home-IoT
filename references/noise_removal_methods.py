# ----------------------------------------------------------------
# Removing noise from time series data can be achieved using various methods, including:

# smoothing techniques,
# filtering techniques, and
# decomposition techniques.
# ----------------------------------------------------------------

"""
Choosing the Right Method for Noise Removal

Choosing the appropriate noise removal method depends on the characteristics of the data and the application goal. For example:

- If the data has a clear periodic pattern, seasonal decomposition can be used.
- If there is a lot of noise in the data, wavelet transform can be tried.
- If real-time smoothing is required, moving average or EWMA can be used.
- If the trend and periodicity need to be preserved, a low-pass filter can be used.

By combining these methods, you can effectively remove noise from time series data and improve the accuracy of anomaly detection and prediction models.
"""

# ----------------------------------------------------------------
# Smoothing Techniques 平滑技术
# 去趋势是为了去除数据中的长期趋势成分，使得数据更加平稳和易于分析；
# 而平滑是为了去除数据中的短期波动，使得数据中的长期趋势更加明显
# ----------------------------------------------------------------
"""
1. Moving Average

Moving average is one of the simplest and most commonly used smoothing methods. It reduces noise by calculating the average of neighboring points in the time series. 
The moving average can be calculated using a fixed window size or a variable window size.
"""
import pandas as pd

# Load the data
data = pd.read_csv("energy_weather_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data.set_index("timestamp", inplace=True)

# Calculate the moving average
data["smoothed"] = data["house_energy"].rolling(window=7).mean()

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data["house_energy"], label="Original")
plt.plot(data["smoothed"], label="Smoothed", color="red")
plt.legend()
plt.show()


"""
2. Exponentially Weighted Moving Average (EWMA)

EWMA is a weighted moving average method that assigns higher weights to more recent observations. 
It is useful for capturing short-term trends and smoothing out noise in the data.
"""

# Calculate the exponentially weighted moving average
data["ewma"] = data["house_energy"].ewm(span=7, adjust=False).mean()

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(data["house_energy"], label="Original")
plt.plot(data["ewma"], label="EWMA", color="red")
plt.legend()
plt.show()

# ----------------------------------------------------------------
# filtering techniques
# ----------------------------------------------------------------

"""
3. Wavelet Transform

Wavelet transform is a powerful tool that can decompose time series data and remove noise. It is effective for handling non-stationary time series.

"""

import pywt  # pip install pywt
import numpy as np


# Using wavelet transform for denoising
def wavelet_denoising(data, wavelet="db4", level=1):
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")


data["denoised"] = wavelet_denoising(data["house_energy"])

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(data["house_energy"], label="Original")
plt.plot(data["denoised"], label="Denoised", color="red")
plt.legend()
plt.show()


"""
4. Low-pass Filter

Low-pass filters can effectively remove high-frequency noise and retain low-frequency signals. The Butterworth filter is a commonly used low-pass filter.

"""

from scipy.signal import butter, filtfilt


# Design the low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Apply the low-pass filter
fs = 1  # Sampling frequency (assuming one data point per minute)
cutoff = 0.1  # Cutoff frequency
data["filtered"] = lowpass_filter(data["house_energy"], cutoff, fs)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(data["house_energy"], label="Original")
plt.plot(data["filtered"], label="Filtered", color="red")
plt.legend()
plt.show()


# ----------------------------------------------------------------
# decomposition techniques
# ----------------------------------------------------------------
"""
5. Seasonal Decomposition

Seasonal decomposition methods decompose a time series into trend, seasonal, and residual components, which can effectively separate and remove noise.

"""

from statsmodels.tsa.seasonal import seasonal_decompose

# Seasonal decomposition
result = seasonal_decompose(data["house_energy"], model="additive", period=365)
data["trend"] = result.trend
data["seasonal"] = result.seasonal
data["residual"] = result.resid

# Visualization
plt.figure(figsize=(10, 6))
plt.subplot(411)
plt.plot(data["house_energy"], label="Original")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(data["trend"], label="Trend")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(data["seasonal"], label="Seasonal")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(data["residual"], label="Residual")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
