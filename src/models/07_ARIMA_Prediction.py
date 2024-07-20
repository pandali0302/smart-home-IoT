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
size = int(len(data) * 0.8)
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


# ----------------------------------------------------------------
# Base Model: MA 移动平均模型
# ----------------------------------------------------------------
# Data resampling by day
train_data_daily = train["use_HO"].resample("d").mean()
test_data_daily = test["use_HO"].resample("d").mean()


ma_model = ARIMA(train_data_daily, order=(0, 0, 10))  # MA model with q=10
ma_model_fit = ma_model.fit()
ma_forecast = ma_model_fit.forecast(steps=len(test_data_daily))
ma_mse, ma_rmse, ma_mae, ma_mape, ma_r2 = evaluate_model(test_data_daily, ma_forecast)

# Print evaluation metrics for each model
print(
    f"MA Model: \nMSE={ma_mse} \nRMSE={ma_rmse} \nMAE={ma_mae} \nMAPE={ma_mape} \nR2={ma_r2}"
)
# MA Model:
# MSE=0.05397983397338097
# RMSE=0.23233560634001188
# MAE=0.18710604764731073
# MAPE=26.373997384222363
# R2=-0.06729702870829368

# plot
plt.figure(figsize=(15, 5))
plt.plot(train_data_daily, label="Train")
plt.plot(test_data_daily, label="Test")
plt.plot(ma_forecast, label="MA Forecast")
plt.legend()
plt.show()


# ----------------------------------------------------------------
# ARIMA 自回归积分滑动平均模型
# ----------------------------------------------------------------
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA

# load and plot data
data_daily = data["use_HO"].resample("d").mean()
# normalize data
data_daily = np.log(data_daily)

rollingMEAN = data_daily.rolling(window=10).mean()
rollingSTD = data_daily.rolling(window=10).std()
# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
plt.subplots_adjust(hspace=0.4)
ax1.plot(data_daily, label="Data - House overall")
ax1.plot(rollingMEAN, label="Rolling mean")
ax2.plot(rollingSTD, c="w", label="Rolling Std")

ax1.legend(fontsize=12), ax2.legend(fontsize=12)
ax1.set_ylabel("kW"), ax2.set_ylabel("kW")
ax1.margins(x=0), ax2.margins(x=0)
ax1.grid(), ax2.grid()


# ----------------------------------------------------------------
# # ADF Test for stationarity
# ----------------------------------------------------------------
resultDFtest = adfuller(data_daily, autolag="AIC")
Out = pd.Series(
    resultDFtest[0:4],
    index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
)
for key, value in resultDFtest[4].items():
    Out["Critical Value (%s)" % key] = value

print("DICK-FULLER RESULTS: \n\n{}".format(Out))

"""
Test Statistic                -5.875177e+00
p-value                        3.171872e-07
#Lags Used                     2.000000e+00
Number of Observations Used    3.480000e+02
Critical Value (1%)           -3.449282e+00
Critical Value (5%)           -2.869881e+00
Critical Value (10%)          -2.571214e+00

- ADF 统计量值越小（越负），说明越有可能拒绝原假设。通常我们会将其与关键值(1%、5% 和 10%)进行比较。
- p 值小于某个显著性水平（例如 0.05），我们可以拒绝原假设，认为数据是平稳的.
- 关键值用于与 ADF 统计量进行比较。如果 ADF 统计量小于（更负于）关键值，则可以拒绝原假设(数据是非平稳的)


结论

根据 ADF 统计量和 p 值，数据是平稳的。因此，可以直接使用这些数据来进行 ARIMA 模型的参数选择和模型训练，而无需进行差分。
"""

# ----------------------------------------------------------------
# Difference the series if needed
# ----------------------------------------------------------------
# If the series is not stationary, difference the series
data_diff = data_daily.diff().dropna()

# Plot the differenced series
plt.figure(figsize=(10, 6))
plt.plot(data_diff, label="Differenced House Energy Consumption")
plt.title("Differenced House Energy Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("Differenced Energy Consumption")
plt.legend()
plt.show()

# ADF Test on differenced series
result = adfuller(data_diff)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
for key, value in result[4].items():
    print("Critical Values:")
    print(f"   {key}, {value}")


# ----------------------------------------------------------------
# # use ACF and PACF plots to select ARIMA model parameters
# ----------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 6))

plt.subplot(121)
plot_acf(data_daily, lags=30, ax=plt.gca())
plt.title("ACF")

plt.subplot(122)
plot_pacf(data_daily, lags=30, ax=plt.gca())
plt.title("PACF")

plt.show()


# ----------------------------------------------------------------
# # Training the ARIMA Model
# ----------------------------------------------------------------
"""
Based on the ACF and PACF plots, we can choose appropriate p and q parameters.

Determining d:

    Since your data is stationary (based on the ADF test), you can set d=0.

Determining p (Autoregressive part):

    Look at the PACF plot. The parameter p is determined by the lag after which the PACF cuts off (i.e., drops close to zero).

    In your PACF plot, the significant spikes diminish after lag 1. Therefore, p=1 could be a good starting point.

Determining q (Moving average part):

    Look at the ACF plot. The parameter q is determined by the lag after which the ACF cuts off.

    In your ACF plot, the significant spikes diminish gradually, suggesting a longer memory in the data. Initially, you might consider q=0 to keep the model simple, and then try higher values based on model performance.
    
"""
p = 1  # AR terms based on PACF plot
d = 0  # Differencing term (0 since data is stationary)
q = 0  # MA terms initial guess based on ACF plot

arima_model = ARIMA(data_daily, order=(p, d, q))
arima_model_fit = arima_model.fit()

print(arima_model_fit.summary())

"""
总结

    模型适合度：模型的AIC和BIC较低，表明模型拟合较好。
    系数显著性：模型中的常数项和自回归项的系数在统计上显著。
    诊断统计量：Ljung-Box Q统计量的p值大于0.05，表明残差没有显著的自相关性，但Jarque-Bera统计量的p值小于0.05，表明残差不服从正态分布，同时存在异方差性。

以上结果表明模型拟合较好，但需要注意残差的非正态性和异方差性。这可能需要进一步处理，例如对数据进行变换（如对数变换）或者使用更复杂的模型来改善残差的特性。

# normalize data based on the fitted model
data_daily = np.log(data_daily) # added it at the beginning
"""


# Forecast
train_data_daily = np.log(train_data_daily)
test_data_daily = np.log(test_data_daily)

forecast_steps = len(test_data_daily)
forecast = arima_model_fit.forecast(steps=forecast_steps)

# Plot the forecast against actual values
plt.figure(figsize=(10, 6))
plt.plot(train_data_daily.index, train_data_daily, label="Training Data")
plt.plot(test_data_daily.index, test_data_daily, label="Test Data", color="orange")
plt.plot(test_data_daily.index, forecast, label="Forecast", color="green")
plt.title("ARIMA Model Forecast")
plt.xlabel("Time")
plt.ylabel("Energy Consumption")
plt.legend()
plt.show()

# Evaluate the ARIMA model
mse, rmse, mae, mape, r2 = evaluate_model(test_data_daily, forecast)
print(f"ARIMA Model: \nMSE={mse} \nRMSE={rmse} \nMAE={mae} \nMAPE={mape} \nR2={r2}")

# ARIMA Model:
# MSE=0.08703779106575858
# RMSE=0.2950216789758993
# MAE=0.22627533192969151
# MAPE=nan
# R2=-0.0696956433335778

# ----------------------------------------------------------------
# # Model Evaluation and Tuning
# ----------------------------------------------------------------
"""
After fitting the initial ARIMA model, evaluate its performance using metrics such as AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), and other evaluation metrics. then iteratively adjust the parameters p and q to improve the model performance.
"""
import itertools

# Define the p, d, and q parameters to take any value between 0 and 3
p = q = range(0, 3)
d = [0]  # Since the data is stationary

# Generate all different combinations of p, d, and q triplets
pdq = list(itertools.product(p, d, q))

# Create a DataFrame to store the results
results = []

# Iterate through all parameter combinations
for param in pdq:
    try:
        model = ARIMA(data_daily, order=param)
        results_ARIMA = model.fit()
        results.append((param, results_ARIMA.aic, results_ARIMA.bic))
    except:
        continue

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["pdq", "AIC", "BIC"])

# Print the top 5 models with the lowest AIC
print(results_df.sort_values(by="AIC").head())

# Choose the best model and fit it
best_param = results_df.sort_values(by="AIC").iloc[0]["pdq"]
best_model = ARIMA(data_daily, order=best_param)
best_model_fit = best_model.fit()

# Print summary of the best model
print(best_model_fit.summary())


# Forecast using the best model
# Rolling forecast
size_daily = int(len(data_daily) * 0.8)
rolling_predictions = []
for i in range(size_daily, len(data_daily)):
    train = data_daily[:i]  # Expanding window
    model = ARIMA(train, order=(1, 0, 0))  # Adjust the order as needed
    model_fit = model.fit()
    pred = model_fit.forecast(steps=1)[0]
    rolling_predictions.append(pred)

# Create a dataframe for comparison
test = data_daily[size_daily:].to_frame()
test["predictions"] = rolling_predictions

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data_daily, label="Original Data")
plt.plot(test["predictions"], label="Rolling Forecast", color="red")
plt.legend()
plt.show()

# Evaluation
mse, rmse, mae, mape, r2 = evaluate_model(test["use_HO"], test["predictions"])
print(f"ARIMA Model: \nMSE={mse} \nRMSE={rmse} \nMAE={mae} \nMAPE={mape} \nR2={r2}")
