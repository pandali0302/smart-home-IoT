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
            原生支持 category 变量，不需要对类别型变量进行独热编码(One-Hot Encoding)，但需要转换为字符串类型；
            支持特征交叉。
        https://github.com/microsoft/LightGBM/tree/master/examples/python-guide


"""

# ----------------------------------------------------------------
# Load Libraries and dataset
# ----------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import shap
# shap.initjs()
import lightgbm as lgb
from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    RandomizedSearchCV,
    GridSearchCV,
)

# ----------------------------------------------------------------
# load data and resample and add time features
# ----------------------------------------------------------------
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

time_features = ["month", "weekday", "day", "timing"]
categorical_features = ["timing"]
target = ["use_HO"]

# 选择特征和目标
features = weather_features + time_features


X = data[features]
y = data[target]


# ----------------------------------------------------------------
# Split data into training and testing sets
# ----------------------------------------------------------------
# 按时间顺序划分数据集
# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# ----------------------------------------------------------------
# LightGBM Model
# ----------------------------------------------------------------
# 创建LightGBM数据集。这种数据格式可以更高效地进行训练，因为它是为LightGBM量身定制的，能够更好地利用内存并加速训练过程。
train_data = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_features
)
test_data = lgb.Dataset(
    X_test, label=y_test, categorical_feature=categorical_features, reference=train_data
)

# 定义参数
params = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": 0,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

# 训练模型  原生的 lgb.train() 接口；
bst = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=True),
        lgb.log_evaluation(10),
    ],
)

# 进行预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
# MSE: 0.4116101904113462

# ----------------------------------------------------------------
# Hyperparameter Tuning
# Random Search
# ----------------------------------------------------------------
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
    objective="regression", metric="mse", boosting_type="gbdt"
)

# 创建TimeSeriesSplit对象
tscv = TimeSeriesSplit(n_splits=5)

# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(
    estimator=lgb_estimator,
    param_distributions=param_grid,
    n_iter=50,
    cv=tscv,
    verbose=0,
    n_jobs=-1,
    random_state=42,
)

# 进行参数调优
random_search.fit(X_train, y_train, categorical_feature=["timing"])

# 打印最佳参数
print(f"Best parameters found by random search are: {random_search.best_params_}")

# 使用最佳参数创建最终模型
best_model = random_search.best_estimator_

# 进行预测
y_final_pred = best_model.predict(X_test)

# 评估模型
final_mse = mean_squared_error(y_test, y_final_pred)
print(f"Final MSE: {final_mse}")


# ----------------------------------------------------------------
# grid search for fine tuning
# ----------------------------------------------------------------
# 基于随机搜索结果进行网格搜索
param_grid_fine = {
    "num_leaves": [
        random_search.best_params_["num_leaves"] - 10,
        random_search.best_params_["num_leaves"],
        random_search.best_params_["num_leaves"] + 10,
    ],
    "learning_rate": [
        random_search.best_params_["learning_rate"] / 2,
        random_search.best_params_["learning_rate"],
        random_search.best_params_["learning_rate"] * 2,
    ],
    "n_estimators": [
        random_search.best_params_["n_estimators"] - 20,
        random_search.best_params_["n_estimators"],
        random_search.best_params_["n_estimators"] + 20,
    ],
    "min_child_samples": [
        random_search.best_params_["min_child_samples"] - 10,
        random_search.best_params_["min_child_samples"],
        random_search.best_params_["min_child_samples"] + 10,
    ],
    "subsample": [
        random_search.best_params_["subsample"] - 0.1,
        random_search.best_params_["subsample"],
        random_search.best_params_["subsample"] + 0.1,
    ],
    "colsample_bytree": [
        random_search.best_params_["colsample_bytree"] - 0.1,
        random_search.best_params_["colsample_bytree"],
        random_search.best_params_["colsample_bytree"] + 0.1,
    ],
}


# 创建LightGBM Regressor
lgb_estimator = lgb.LGBMRegressor(
    objective="regression", metric="mse", boosting_type="gbdt"
)

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=lgb_estimator, param_grid=param_grid_fine, cv=tscv, verbose=0, n_jobs=-1
)

# 进行参数调优
grid_search.fit(X_train, y_train, categorical_feature=["timing"])

# 打印最佳参数
print(f"Best parameters found by grid search are: {grid_search.best_params_}")

# 使用最佳参数创建最终模型
best_model = grid_search.best_estimator_

# 进行预测
y_final_pred = best_model.predict(X_test)

# 评估模型
final_mse = mean_squared_error(y_test, y_final_pred)
print(f"Final MSE: {final_mse}")


# ----------------------------------------------------------------
# find best parameters and train the final model
# ----------------------------------------------------------------
best_params = grid_search.best_params_
# 更新参数
params.update(best_params)

# 创建LightGBM数据集
train_data = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_features
)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


# to record eval results for plotting
evals_result = {}

# 训练最终模型
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=[
        # lgb.early_stopping(stopping_rounds=20, verbose=True),
        lgb.log_evaluation(10),
        lgb.record_evaluation(evals_result),
    ],
)

# 进行预测
y_final_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# 评估模型
final_mse = mean_squared_error(y_test, y_final_pred)
print(f"Final MSE: {final_mse}")

# ----------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(evals_result["train"]["l2"], label="Training Loss")
plt.plot(evals_result["test"]["l2"], label="Validation Loss")
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.show()

# ----------------------------------------------------------------
# 自定义损失函数和评估函数
# https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Plot feature importances
# https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/notebooks/interactive_plot_example.ipynb
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# save model
# ----------------------------------------------------------------
# best_model.save_model("../../models/lgb_model.txt")

import joblib

ref_cols = list(X.columns)

"""
In Python, you can use joblib or pickle to serialize (and deserialize) an object structure into (and from) a byte stream. 
In other words, it's the process of converting a Python object into a byte stream that can be stored in a file.

https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html

"""

joblib.dump(value=[best_model, ref_cols, target], filename="../../models/model.pkl")

# Load the model from a file
lgb_model_loaded = joblib.load("../../models/lightgbm_model.pkl")
