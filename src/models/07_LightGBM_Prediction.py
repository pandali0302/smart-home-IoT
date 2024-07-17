# ----------------------------------------------------------------
# Modeling for Energy Consumption Prediction
# ----------------------------------------------------------------
"""
- 传统时间序列预测模型,像MA、VAR和ARIMA,理论基础扎实,计算效率高,如果是处理单变量的预测问题，传统时序模型可以发挥较大的优势；但在处理复杂的非线性关系和多变量交互效应方面,就显得有点力不从心。

- 深度学习模型,如LSTM、DARNN、DeepGlo、TFT和DeepAR,自动学习数据中的复杂模式和特征,在多个预测任务中展示出强大的性能。

- GBRT模型,以lightgbm、xgboost 为代表，一般就是把时序问题转换为监督学习，通过特征工程和机器学习方法去预测；
    这种模型可以解决绝大多数的复杂的时序预测模型。支持复杂的数据建模，支持多变量协同回归，支持非线性问题。
    在实验中表现优越,尤其在适当配置的情况下,能够超过许多最先进的深度学习模型。
    不过这种方法需要较为复杂的人工特征过程部分，需要一定的经验和技巧。

- 特征工程和损失函数,在机器学习中至关重要,合理的特征设计和损失函数选择能够显著提升模型性能。

- 模型架构的创新带来的提升有限，优先关注特征工程和损失函数的优化更为重要。

机器学习模型：适合处理复杂特征和非线性关系
    1. SVM  
        --  对数据的尺度敏感，特别是使用欧几里得距离作为核函数时，通常需要标准化
            在高维空间中非常有效
    2. LightGBM 
        -- 基于梯度提升的决策树算法，对特征的尺度不是非常敏感。不一定需要对数据进行归一化或标准化
            计算速度快，模型精度高；
            缺失值不需要处理，比较方便；
            支持 category 变量；
            支持特征交叉。

    3. XGBoost

"""

# ----------------------------------------------------------------
# Load Libraries and dataset
# ----------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# import shap
# shap.initjs()
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


df = pd.read_pickle("../../data/interim/02_add_time_features.pkl")
data = df.copy()

# ----------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------
"""
use weather information and time information as regression features during prediction
"""
#  time features extraction like month, day, hour, weekday, etc. -- done
#  weather features extraction like temperature, humidity, wind speed, etc.
#  interaction features between weather conditions and energy consumption
#  lag features to capture temporal dependencies
#  rolling window statistics to capture temporal patterns
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

time_features = ["month", "weekday", "day"]

target = ["use_HO"]

# 对类别变量timing列进行独热编码
data = pd.get_dummies(data, columns=["timing"])

# 选择特征和目标
features = weather_features + time_features
features += [
    col for col in data.columns if "timing_" in col
]  # 添加独热编码后的timing列

X = data[features]
y = data[target]


# ----------------------------------------------------------------
# Split data into training and testing sets
# ----------------------------------------------------------------
# 按时间顺序划分数据集
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# ----------------------------------------------------------------
# LightGBM Model
# ----------------------------------------------------------------
# 创建LightGBM数据集。这种数据格式可以更高效地进行训练，因为它是为LightGBM量身定制的，能够更好地利用内存并加速训练过程。
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 定义参数
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": 0,
}

# 训练模型  原生的 lgb.train() 接口；
bst = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=True),
        lgb.log_evaluation(),
    ],
)

# 进行预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# 评估模型
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# ----------------------------------------------------------------
# Hyperparameter Tuning
# Random Search
# ----------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# 定义参数空间
param_grid = {
    "num_leaves": [31, 50, 70, 100],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500, 1000],
    "min_child_samples": [20, 30, 40, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# 创建LightGBM Regressor   sklearn 的 lgb.LGBMRegressor() 接口
lgb_estimator = lgb.LGBMRegressor(
    objective="regression", metric="rmse", boosting_type="gbdt"
)

# 创建TimeSeriesSplit对象
tscv = TimeSeriesSplit(n_splits=5)

# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(
    estimator=lgb_estimator,
    param_distributions=param_grid,
    n_iter=50,
    cv=tscv,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

# 进行参数调优
random_search.fit(X_train, y_train)

# 打印最佳参数
print(f"Best parameters found by random search are: {random_search.best_params_}")
# RMSE: 0.6441369430176253

# ----------------------------------------------------------------
# find best parameters and train the final model
# ----------------------------------------------------------------
best_params = random_search.best_params_

# 更新参数
params.update(best_params)

# 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 记录训练过程的评估结果
evals_result = {}

# 训练最终模型
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=True),
        lgb.log_evaluation(),
        lgb.record_evaluation(evals_result),
    ],
)

# 进行预测
y_final_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# 评估模型
final_rmse = mean_squared_error(y_test, y_final_pred, squared=False)
print(f"Final RMSE: {final_rmse}")
# Final RMSE: 0.6455591879731212


# ----------------------------------------------------------------
# plot loss curve
# ----------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(evals_result["train"]["rmse"], label="Train RMSE")
plt.plot(evals_result["test"]["rmse"], label="Test RMSE")
plt.xlabel("Number of Iterations")
plt.ylabel("RMSE")
plt.title("Training and Validation RMSE over Iterations")
plt.legend()
plt.show()

# ----------------------------------------------------------------
# plot
# ----------------------------------------------------------------
df = pd.DataFrame(
    {
        "test": y_test.values.flatten(),
        "Predicted": y_final_pred.flatten(),
    },
    index=y_test.index,
)


# 绘制真实值和预测值的折线图
plt.figure(figsize=(10, 6))
plt.plot(df["test"], label="Test")
plt.plot(df["Predicted"], label="Predicted")
plt.xlabel("Time")
plt.ylabel("Energy Consumption")
plt.title("True vs Predicted Energy Consumption")
plt.legend()
plt.show()
