# ----------------------------------------------------------------
# import libraries
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
directory = "../../data/raw/chunks"

# 使用 os.walk 遍历目录
for dirpath, dirnames, filenames in os.walk(directory):
    print(f"Directory: {dirpath}")
    for filename in filenames:
        print(f"File: {os.path.join(dirpath, filename)}")

files = glob("../../data/raw/chunks/*.csv")
len(files)
for f in files:
    df = pd.read_csv(f)

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


# loop through each column and check the unique values
for col in categorical_cols:
    print(f"Column: {col}")
    print(df[col].unique())
    print("\n")


# check the percentage of missing values
df.isnull().mean()

# check duplicate rows
df.duplicated().sum()


# --------------------------------------------------------------
# Normal Distribution  Plot
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# plotting settings
plt.style.available
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100


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
plot_df = df[["use [kW]"]]
plot_df.head()

UVA_numeric(plot_df)


# ----------------------------------------------------------------
# Data Pre-processing
# ----------------------------------------------------------------
# remove unit(kW) from dataset column names.
df.columns = df.columns.str.replace(" \[kW\]", "", regex=True)


# aggregat some columns into new column by summing them up.
df["Furnace"] = df[["Furnace 1", "Furnace 2"]].sum(axis=1)
df["Kitchen"] = df[["Kitchen 12", "Kitchen 14", "Kitchen 38"]].sum(axis=1)

df.drop(
    columns=["Furnace 1", "Furnace 2", "Kitchen 12", "Kitchen 14", "Kitchen 38"],
    inplace=True,
)


# deal with column time and convert it to datetime
df["time"] = pd.to_datetime(df["time"], unit="s")
df["time"].head()
df["time"] = pd.DatetimeIndex(
    pd.date_range("2016-01-01 05:00", periods=len(df), freq="min")
)

# To utilize datetime information such as year, month and day in EDA and modeling phase, we need to extract them from time column.
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.weekday
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour
df["minute"] = df["time"].dt.minute

#  based on column hour, add a new column to indicate the different period (Night, Morning, Afternoon and Evening ) of a day.

#     Night : 22:00 - 23:59 / 00:00 - 03:59
#     Morning : 04:00 - 11:59
#     Afternoon : 12:00 - 16:59
#     Evening : 17:00 - 21:59

# We can create timing variable based on hour variable.
df["hour"].unique()
df["timing"] = "Night"
# df.loc[(df["hour"] >= 22) & (df["hour"] < 24), "timing"] = "Night"
df.loc[(df["hour"] >= 4) & (df["hour"] < 12), "timing"] = "Morning"
df.loc[(df["hour"] >= 12) & (df["hour"] < 17), "timing"] = "Afternoon"
df.loc[(df["hour"] >= 17) & (df["hour"] < 22), "timing"] = "Evening"
df.head()

df.info()


# Replace invalid values in column 'cloudCover' with backfill method
df["cloudCover"].replace(["cloudCover"], method="bfill", inplace=True)
df["cloudCover"] = df["cloudCover"].astype("float")


# drop columns that are not useful for analysis
df["icon"].value_counts()
df = df.drop(columns=["summary", "icon"])

# Removing Duplicate Columns


# --------------------------------------------------------------
# EDA
# --------------------------------------------------------------
# check the correlation between continuous variables
corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
