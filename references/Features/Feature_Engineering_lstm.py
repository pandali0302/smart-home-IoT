"""
在构建多变量 LSTM 模型时，特征工程是提高预测准确性的关键步骤之一。以下是一些常见的特征工程技术：

1. 特征选择：
   - 确定哪些变量对预测任务最为重要，并选择这些特征进行建模。

2. 特征缩放：
   - 对特征进行标准化或归一化处理，以消除不同量纲和量级带来的影响。

3. 特征构造：
   - 创建新的特征，例如时间延迟特征（时间序列的滞后值）、滚动窗口统计特征（如移动平均、标准差等）。

4. 特征转换：
   - 对特征应用数学转换，如对数转换、Box-Cox转换等，以稳定方差或使数据更符合正态分布。

5. 缺失值处理：
   - 填充或插值缺失的观测值，或使用模型能够处理缺失值的特性。

6. 异常值处理：
   - 识别并处理异常值，以防止它们对模型训练产生不良影响。

7. 特征交互：
   - 构造特征间的交互项，以捕捉变量之间的相互作用。

8. 时间特征：
   - 利用时间相关的特征，如小时、星期、月份等，这些特征可以提供季节性信息。

9. 周期性特征：
   - 对于具有周期性的数据，可以构造周期性特征来帮助模型学习时间序列的周期性规律。

10. 差分特征：
    - 对于非平稳时间序列，构造差分特征以帮助模型捕捉趋势和季节性。

11. 外部数据融合：
    - 将外部数据源的信息融合到模型中，如经济指标、天气情况等，以提供额外的预测信息。

12. 特征降维：
    - 使用 PCA（主成分分析）或 t-SNE 等方法对高维特征进行降维，以减少模型的复杂度。

13. 特征编码：
    - 对类别型变量进行编码，如使用独热编码（One-Hot Encoding）。

14. 数据重采样：
    - 如果数据的时间粒度不一致，可能需要重采样以确保数据的一致性。

15. 数据分割：
    - 将时间序列数据分割成适合模型训练的序列，例如使用滑动窗口方法。
"""
### 示例代码

import pandas as pd
import numpy as np

# 假设 df 是包含多变量时间序列数据的 DataFrame

# 特征缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 构造时间延迟特征
for i in range(1, 4):
    df[f'feature_lag_{i}'] = df['feature'].shift(i)

# 构造滚动窗口统计特征
df['feature_rolling_mean'] = df['feature'].rolling(window=5).mean()
df['feature_rolling_std'] = df['feature'].rolling(window=5).std()

# 对数转换
df['feature_log'] = np.log1p(df['feature'])  # 使用 log1p 以处理 0 值

# 填充缺失值
df.fillna(method='ffill', inplace=True)  # 前向填充

# PCA 降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df.drop(['feature'], axis=1)),
                      columns=['pca1', 'pca2'])
```
