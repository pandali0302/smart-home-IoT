# ----------------------------------------------------------------
# import libraries
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
# pandas.read_csv: Read a comma-separated values (csv) file into DataFrame.
file_path = "../../data/raw/subset_file.csv"
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

# ----------------------------------------------------------------
# Data Pre-processing
# ----------------------------------------------------------------
# deldete last row (NaN values)
df = df[:-1]

# remove unit(kW) from dataset column names.
df.columns = df.columns.str.replace(" \[kW\]", "", regex=True)


# aggregat some columns into new column by summing them up.
df["Furnace"] = df[["Furnace 1", "Furnace 2"]].sum(axis=1)
df["Kitchen"] = df[["Kitchen 12", "Kitchen 14", "Kitchen 38"]].sum(axis=1)

df.drop(
    columns=["Furnace 1", "Furnace 2", "Kitchen 12", "Kitchen 14", "Kitchen 38"],
    inplace=True,
)


# Replace invalid values in column 'cloudCover' with backfill method
df["cloudCover"].value_counts()
df["cloudCover"].replace(["cloudCover"], method="bfill", inplace=True)
df["cloudCover"] = df["cloudCover"].astype("float")


# drop columns that are not useful for analysis
df["icon"].value_counts()
df = df.drop(columns=["summary", "icon"])

df_basic = df
df_TimeSeries = df.set_index("time")


# Check Correlation between numerical columns
fig = plt.subplots(figsize=(10, 8))
corr = df_TimeSeries[numerical_cols].corr()
sns.heatmap(corr[corr > 0.9], vmax=1, vmin=-1, center=0)
plt.show()


#'use' - 'House overall'
# 'gen' and 'Solar'
# 'temperature' and 'apparentTemperature'
# They are indeed the same data (overlaping perfectly)
fig, axes = plt.subplots(3, 1, figsize=(10, 5))
df_TimeSeries[["use", "House overall"]].resample("D").mean().plot(ax=axes[0])
df_TimeSeries[["gen", "Solar"]].resample("D").mean().plot(ax=axes[1])
df_TimeSeries[["temperature", "apparentTemperature", "dewPoint"]].resample(
    "D"
).mean().plot(ax=axes[2])


# columns' correlation coefficient is almost over 0.95, so we need to put these columns together as a new columns.
df_TimeSeries[["temperature", "apparentTemperature", "dewPoint"]].corr()

# Removing Duplicate Columns
df_TimeSeries["use_HO"] = df_TimeSeries["use"]
df_TimeSeries["gen_Solar"] = df_TimeSeries["gen"]
df_TimeSeries.drop(
    ["use", "House overall", "gen", "Solar", "apparentTemperature"],
    axis=1,
    inplace=True,
)
df_TimeSeries.columns

# rearrange columns
df_TimeSeries = df_TimeSeries[
    [
        "use_HO",
        "gen_Solar",
        "Furnace",
        "Kitchen",
        "Dishwasher",
        "Home office",
        "Fridge",
        "Wine cellar",
        "Garage door",
        "Barn",
        "Well",
        "Microwave",
        "Living room",
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
]
df_TimeSeries.head()
df_TimeSeries.info()


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

file_path = "../../data/raw/subset_file.csv"


def read_preprocessing_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    # deldete last row (NaN values)
    df = df[:-1]

    # remove unit(kW) from dataset column names.
    df.columns = df.columns.str.replace(" \[kW\]", "", regex=True)

    # aggregat some columns into new column by summing them up.
    df["Furnace"] = df[["Furnace 1", "Furnace 2"]].sum(axis=1)
    df["Kitchen"] = df[["Kitchen 12", "Kitchen 14", "Kitchen 38"]].sum(axis=1)

    df.drop(
        columns=["Furnace 1", "Furnace 2", "Kitchen 12", "Kitchen 14", "Kitchen 38"],
        inplace=True,
    )

    # Replace invalid values in column 'cloudCover' with backfill method
    df["cloudCover"].replace(["cloudCover"], method="bfill", inplace=True)
    df["cloudCover"] = df["cloudCover"].astype("float")

    # drop columns that are not useful for analysis
    df = df.drop(columns=["summary", "icon"])

    df = df.set_index("time")

    # Removing Duplicate Columns
    df["use_HO"] = df["use"]
    df["gen_Solar"] = df["gen"]
    df.drop(
        ["use", "House overall", "gen", "Solar", "apparentTemperature"],
        axis=1,
        inplace=True,
    )

    # rearrange columns
    df = df[
        [
            "use_HO",
            "gen_Solar",
            "Furnace",
            "Kitchen",
            "Dishwasher",
            "Home office",
            "Fridge",
            "Wine cellar",
            "Garage door",
            "Barn",
            "Well",
            "Microwave",
            "Living room",
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
    ]

    return df


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_TimeSeries.to_pickle("../../data/interim/01_data_processed.pkl")
