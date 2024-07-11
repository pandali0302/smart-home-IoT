# ----------------------------------------------------------------
# import libraries
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime

from glob import glob
import os
import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# ----------------------------------------------------------------
# Load datasets in dircetory
# ----------------------------------------------------------------
# 指定要遍历的目录
directory = "../../data/raw"

# 使用 os.walk 遍历目录
for dirpath, dirnames, filenames in os.walk(directory):
    print(f"Directory: {dirpath}")
    for filename in filenames:
        print(f"File: {os.path.join(dirpath, filename)}")


# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
# pandas.read_csv: Read a comma-separated values (csv) file into DataFrame.
file_path = "../../data/raw/chunks/HomeC_split_1.csv"
df = pd.read_csv(file_path, low_memory=False)

df.shape

df.head()
df.tail()
# ----------------------------------------------------------------
# Data Exploration
# ----------------------------------------------------------------
df.info(verbose=True)
df.describe(include=["object"])
df.describe()  # include = ‘all’

df.select_dtypes(include=["float64"]).columns
df.select_dtypes(include=["object"]).columns


categorical_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
numerical_cols = [name for name in df.columns if df[name].dtype in ["int64", "float64"]]

# check unique values in each column
unique_values = [df[col].unique() for col in categorical_cols]


# check the percentage of missing values
df.isnull().mean()

# check duplicate rows
df.duplicated().sum()


# --------------------------------------------------------------
# Normal Distribution
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def UVA_numeric(data):
    var_group = data.columns
    size = len(var_group)
    plt.figure(figsize=(7 * size, 3), dpi=400)

    # looping for each variable
    for j, i in enumerate(var_group):
        # calculating descriptives of variable
        mini = data[i].min()
        maxi = data[i].max()
        ran = maxi - mini
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        # calculating points of standard deviation
        points = mean - st_dev, mean + st_dev

        # Plotting the variable with every information
        plt.subplot(1, size, j + 1)
        sns.histplot(data[i], kde=True)

        sns.lineplot(x=points, y=[0, 0], color="black", label="std_dev")
        sns.scatterplot(x=[mini, maxi], y=[0, 0], color="orange", label="min/max")
        sns.scatterplot(x=[mean], y=[0], color="red", label="mean")
        sns.scatterplot(x=[median], y=[0], color="blue", label="median")
        plt.xlabel("{}".format(i), fontsize=20)
        plt.ylabel("density")
        plt.title(
            "std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}".format(
                round(points[0], 2), round(points[1], 2), skew, ran, mean, median
            )
        )

plot_df = df[["use [kW]", "gen [kW]", "House overall [kW]"]]
plot_df = df[["gen [kW]"]]
plot_df.head()

UVA_numeric(plot_df)


# ----------------------------------------------------------------
# Data Pre-processing
# ----------------------------------------------------------------

# deal with column time and convert it to datetime
df["time"] = pd.to_datetime(df["time"])

# clean column names
# drop na


# --------------------------------------------------------------
# EDA
# --------------------------------------------------------------
# check the correlation between continuous variables
corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
