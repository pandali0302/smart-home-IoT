"""
- 传统时间序列预测模型,像MA、VAR和ARIMA,理论基础扎实,计算效率高,如果是处理单变量的预测问题，传统时序模型可以发挥较大的优势；但在处理复杂的非线性关系和多变量交互效应方面,就显得有点力不从心。

- 深度学习模型,如LSTM、DARNN、DeepGlo、TFT和DeepAR,自动学习数据中的复杂模式和特征,在多个预测任务中展示出强大的性能。

- GBRT模型,以lightgbm、xgboost 为代表，一般就是把时序问题转换为监督学习，通过特征工程和机器学习方法去预测；
    这种模型可以解决绝大多数的复杂的时序预测模型。支持复杂的数据建模，支持多变量协同回归，支持非线性问题。
    在实验中表现优越,尤其在适当配置的情况下,能够超过许多最先进的深度学习模型。
    不过这种方法需要较为复杂的人工特征过程部分，需要一定的经验和技巧。

- 特征工程和损失函数,在机器学习中至关重要,合理的特征设计和损失函数选择能够显著提升模型性能。



机器学习模型：适合处理复杂特征和非线性关系
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
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

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
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
data = df.copy()
data = data.resample("D").mean()
data.info()
data.head(20)
data.tail()

data["year"] = data.index.year
data["quarter"] = data.index.quarter
data["month"] = data.index.month
data["weekday"] = data.index.weekday


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

time_features = ["year", "quarter", "month", "weekday"]
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 可视化训练集和测试集之间的分割:
plt.figure(figsize=(15, 5))
plt.plot(X_train.index, y_train, label="train")
plt.plot(X_test.index, y_test, label="test")
plt.legend()
plt.show()


# ----------------------------------------------------------------
# Model training
# ----------------------------------------------------------------
# 定义参数空间
param_grid = {
    "num_leaves": [20, 30, 50, 70, 100],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500, 1000],
    "min_child_samples": [20, 30, 40, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
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
    verbose=-1,
    n_jobs=-1,
    random_state=42,
)

# 进行参数调优
random_search.fit(X_train, y_train)

# 打印最佳参数
print(f"Best parameters found by random search are: {random_search.best_params_}")


# 使用最佳参数创建最终模型
best_model = random_search.best_estimator_

# 进行预测
y_final_pred = best_model.predict(X_test)


# ----------------------------------------------------------------
# evaluate model
# ----------------------------------------------------------------
def evaluate_model(y_test, prediction):
    print(f"MAE: {mean_absolute_error(y_test, prediction)}")
    print(f"MSE: {mean_squared_error(y_test, prediction)}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, prediction)}")


evaluate_model(y_test, y_final_pred)
# MAE: 0.17146917462398556
# MSE: 0.04947370236449221
# MAPE: 0.22951007822761418

# ----------------------------------------------------------------
# Plot predictions
# ----------------------------------------------------------------
plt.figure(figsize=(15, 5))
# plt.plot(X.index, y, label="actual")
plt.plot(X_test.index, y_test, label="actual")
plt.plot(X_test.index, y_final_pred, label="predicted")
plt.legend()
plt.show()


# ----------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------
# Plot feature importance,  for lgb.Booster object 原生接口
plt.figure(figsize=(10, 6))
lgb.plot_importance(best_model, max_num_features=10)
plt.title("Feature Importance")
plt.show()

#  for 'LGBMRegressor' object
plt.figure(figsize=(10, 6))
plt.barh(X_train.columns, best_model.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# Get feature importance values
importance = best_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importance}
)
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
)

print(feature_importance_df)

# ----------------------------------------------------------------
# save model
# ----------------------------------------------------------------
import joblib

joblib.dump(best_model, "../../models/lightgbm_model.pkl")


# Save the model -- only for lgb.Booster object
model_filename = "lightgbm_model.txt"
best_model.save_model("../../models/" + model_filename)

# Load the model from a file
loaded_model = lgb.Booster("../../models/" + model_filename)

# Make predictions with the loaded model
y_loaded_pred = loaded_model.predict(X_test, num_iteration=loaded_model.best_iteration)
