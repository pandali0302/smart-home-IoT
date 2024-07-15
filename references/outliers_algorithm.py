import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from sklearn.neighbors import LocalOutlierFactor
import math
import scipy

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
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution
df[outlier_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
df[outlier_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)


# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# --------------------------------------------------------------
# Local outlier factor (density based)
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
col = "Fridge"
dataset, _ = mark_outliers_isolation_forest(df, [col])
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col="outlier_iso_forest", reset_index=True
)


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col=col)
dataset.loc[dataset["gyr_z_outlier"], col] = np.nan


# Create a loop

outlier_removed_df = df.copy()

for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)

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

outlier_removed_df.to_pickle("../../data/interim/02.pkl")
