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
file_path = "../../data/raw/HomeC.csv"
df = pd.read_csv(file_path, low_memory=False)

df.head()
df.tail()
# ----------------------------------------------------------------
# Data Exploration
# ----------------------------------------------------------------
df.info()
df.describe(include=["object"])

df.select_dtypes(include=["float64"]).columns

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
