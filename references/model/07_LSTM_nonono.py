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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----------------------------------------------------------------
# load the dataset
# ----------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_processed.pkl")

data = df.copy()

# 重采样为每日数据
daily_data = data.resample("D").mean()


# ----------------------------------------------------------------
# Split data into training and testing sets
# ----------------------------------------------------------------
train_size = int(len(daily_data) * 0.7)
train_data = daily_data[:train_size]
test_data = daily_data[train_size:]

# ----------------------------------------------------------------
# feature scaling
# ----------------------------------------------------------------

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


# 滑动窗口包含待预测特征
def prepare_data(data, win_size, target_feature_idx):
    num_features = data.shape[1]
    X = []
    y = []
    for i in range(len(data) - win_size):
        temp_x = data.iloc[i : i + win_size, :]
        temp_y = data.iloc[i + win_size, target_feature_idx]
        X.append(temp_x)
        y.append(temp_y)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


win_size = 1  # 时间窗口
target_feature_idx = 0  # 指定待预测特征列
X_train, y_train = prepare_data(scaled_train_df, win_size, target_feature_idx)
X_test, y_test = prepare_data(scaled_test_df, win_size, target_feature_idx)
print("训练集形状:", X_train.shape, y_train.shape)  # (244, 1, 11) (244,)
print("测试集形状:", X_test.shape, y_test.shape)

# ----------------------------------------------------------------
# Build the LSTM model
# ----------------------------------------------------------------
model = Sequential()
model.add(
    LSTM(
        25,
        activation="relu",
        return_sequences=False,
        input_shape=(X_train.shape[1], X_train.shape[2]),
    )
)
# model.add(LSTM(25, activation='relu', return_sequences = False))
# model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.summary()


# 定义回调
checkpoint_cb = ModelCheckpoint(
    "../../models/LSTM/model_checkpoint.h5", save_best_only=True
)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

# 训练模型，并使用回调
history = model.fit(
    X_train,
    y_train,
    epochs=100,  # 设定一个较大的最大训练轮数
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# 可视化训练和验证损失
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# 调用函数绘制损失曲线
plot_loss(history)


# ----------------------------------------------------------------
# predict the test data
# ----------------------------------------------------------------
# 使用训练好的模型对测试数据集进行预测
y_pred = model.predict(X_test)

# Inverse scaling
y_pred_unscaled = scaler.inverse_transform(
    np.concatenate([np.zeros((y_pred.shape[0], X_test.shape[2] - 1)), y_pred], axis=1)
)[:, -1]

y_test_unscaled = scaler.inverse_transform(
    np.concatenate(
        [np.zeros((y_test.shape[0], X_test.shape[2] - 1)), y_test.reshape(-1, 1)],
        axis=1,
    )
)[:, -1]


# 可视化预测结果和真实值
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(15, 5))
    plt.plot(y_true, label="True Values")
    plt.plot(y_pred, label="Predicted Values")
    plt.title("True Values vs Predicted Values")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()


# 调用函数绘制预测结果和真实值
plot_predictions(y_test_unscaled, y_pred_unscaled)
plot_predictions(y_test, y_pred)

# ----------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------

# 计算评估指标
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
r2 = r2_score(y_test_unscaled, y_pred_unscaled)

# 输出评估指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# Mean Squared Error (MSE): 0.058325971839282335
# Root Mean Squared Error (RMSE): 0.2415077055484614
# Mean Absolute Error (MAE): 0.1912015371940518
# R^2 Score: -0.25607121625848217


# ----------------------------------------------------------------
# save the model
# ----------------------------------------------------------------
# 保存模型到文件
model.save("../../models/LSTM/my_lstm_model.h5")


# ----------------------------------------------------------------
# load the model
# ----------------------------------------------------------------
# 加载模型
loaded_model = load_model("../../models/LSTM/my_lstm_model.h5")
loaded_model.summary()


# 使用加载的模型进行预测（与之前的预测步骤相同）
loaded_model_predictions = loaded_model.predict(X_test)

# 反归一化预测结果
loaded_model_predictions_unscaled = scaler.inverse_transform(
    np.concatenate(
        [
            np.zeros((loaded_model_predictions.shape[0], X_test.shape[2] - 1)),
            loaded_model_predictions,
        ],
        axis=1,
    )
)[:, -1]


# ----------------------------------------------------------------
# Anomaly Detection
# ----------------------------------------------------------------
# 使用训练好的模型对测试数据集进行预测
y_pred = model.predict(X_test)

# 反归一化预测值和真实值
y_test_unscaled = scaler.inverse_transform(
    np.concatenate(
        [np.zeros((y_test.shape[0], X_test.shape[2] - 1)), y_test.reshape(-1, 1)],
        axis=1,
    )
)[:, -1]
y_pred_unscaled = scaler.inverse_transform(
    np.concatenate([np.zeros((y_pred.shape[0], X_test.shape[2] - 1)), y_pred], axis=1)
)[:, -1]

# 计算预测误差
errors = np.abs(y_test_unscaled - y_pred_unscaled)

# 设置异常检测的阈值（这里我们使用3倍的标准差作为阈值）
threshold = 3 * np.std(errors)

# 标记异常
anomalies = errors > threshold

# 计算上下界
upper_bound = y_pred_unscaled + threshold
lower_bound = y_pred_unscaled - threshold


# 可视化真实值、预测值、异常点和上下界
def plot_anomalies_with_bounds(y_true, y_pred, anomalies, upper_bound, lower_bound):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label="True Values")
    plt.plot(y_pred, label="Predicted Values")
    plt.fill_between(
        range(len(y_pred)),
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.2,
        label="Prediction Bound",
    )
    plt.scatter(
        np.where(anomalies)[0], y_true[anomalies], color="red", label="Anomalies"
    )
    plt.title("True Values vs Predicted Values with Anomalies and Prediction Bound")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()


# 调用函数绘制异常点和上下界
plot_anomalies_with_bounds(
    y_test_unscaled, y_pred_unscaled, anomalies, upper_bound, lower_bound
)
