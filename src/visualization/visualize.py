import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Normal Distribution  Plot
# --------------------------------------------------------------

# plotting settings
plt.style.available
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100


def UVA_numeric(data):
    var_group = data.columns

    # Looping for each variable
    for i in var_group:
        plt.figure(figsize=(10, 5), dpi=100)

        # Calculating descriptives of variable
        mini = data[i].min()
        maxi = data[i].max()
        ran = maxi - mini
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        # Calculating points of standard deviation
        points = mean - st_dev, mean + st_dev

        # Plotting the variable with every information
        sns.histplot(data[i], kde=True)

        sns.lineplot(x=points, y=[0, 0], color="black", label="std_dev")
        sns.scatterplot(x=[mini, maxi], y=[0, 0], color="orange", label="min/max")
        sns.scatterplot(x=[mean], y=[0], color="red", label="mean")
        sns.scatterplot(x=[median], y=[0], color="blue", label="median")
        plt.xlabel(i, fontsize=12)
        plt.ylabel("density")
        plt.title(
            "std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}".format(
                round(points[0], 2), round(points[1], 2), skew, ran, mean, median
            ),
            fontsize=10,
        )
        plt.legend()
        plt.show()


plot_df = df[["use_HO", "gen_Solar", "Dishwasher"]]

UVA_numeric(plot_df)
