from matplotlib.pylab import f
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.api.types import CategoricalDtype

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# --------------------------------------------------------------
# Normal Distribution  Plot
# --------------------------------------------------------------
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

        # Set consistent x and y limits
        plt.xlim(mini - 0.1 * ran, maxi + 0.1 * ran)
        plt.ylim(bottom=0)

        plt.xlabel(i, fontsize=12)
        plt.ylabel("density")
        plt.title(
            "std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}".format(
                round(points[0], 2), round(points[1], 2), skew, ran, mean, median
            ),
            fontsize=10,
        )
        plt.legend(loc="upper right")
        plt.show()


plot_df = df[["use_HO", "gen_Solar"]]
plot_df = df[["Microwave"]]

UVA_numeric(plot_df)

df["Microwave"].describe()


# ----------------------------------------------------------------
# Correlation Analysis
# ----------------------------------------------------------------
def correlation_plot(data):
    plt.figure(figsize=(15, 8), dpi=100)
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Plot")
    plt.show()


df.info()
app_columns = df.columns[:13]
weather_columns = df.columns[13:]

correlation_plot(df[app_columns])
# no corelation between appliances, use_HO is hightly correlated with Furnace and Dishwasher
correlation_plot(df[weather_columns])  # dewPoint and temperature are highly correlated
correlation_plot(df[app_columns.append(weather_columns)])
# Some appliances (Fridge and Wine cellar and Furnace) are affected by weather information


# ----------------------------------------------------------------
# Identify Peak Usage Times
# ----------------------------------------------------------------

# Identify Peak Usage Times: Determine when each appliance consumes the most energy.
# Resample the data to hourly intervals and sum the consumption
hourly_data = df.groupby("hour")[app_columns].sum()
# Find the hour with the maximum consumption for each appliance
peak_hours = hourly_data.idxmax()

print("Peak usage hours for each appliance:")
print(peak_hours)

# ----------------------------------------------------------------
# Correlate with Activities
# ----------------------------------------------------------------
# Correlate with Activities: Relate energy spikes to specific activities (e.g., cooking, working in the home office).
cooking_hourly_data = (
    hourly_data["Kitchen"] + hourly_data["Dishwasher"] + hourly_data["Microwave"]
)
working_hourly_data = hourly_data["use_HO"]
cooking_peak_hours = cooking_hourly_data.idxmax()
working_peak_hours = working_hourly_data.idxmax()

print(f"Peak cooking usage hours: {cooking_peak_hours}")  # 1
print(f"Peak working usage hours: {working_peak_hours}")  # 23


# ----------------------------------------------------------------
# Time-Series Analysis / Seasonality and Trend
# ----------------------------------------------------------------
# Iterate each appliance, plot the time series, resample to daily
def time_series_plot(data):
    data_resampled = data.resample(
        "D"
    ).mean()  # Resample time to day and calculate the mean
    for i in data_resampled.columns:
        plt.figure(figsize=(15, 8), dpi=100)
        sns.lineplot(x=data_resampled.index, y=data_resampled[i], label=i)
        plt.title(f"{i} Time Series")
        plt.legend()
        plt.show()


time_series_plot(df[app_columns])
#  some appliances have seasonality, some have trend, some have both
#  Furnace has a trend, Wine cellar has seasonality, Fridge has both

time_series_plot(df[weather_columns])
#  temperature and dewPoint have seasonality, visibility has trend


# ----------------------------------------------------------------
# Time feature extraction
# ----------------------------------------------------------------
# To utilize datetime information such as year, month and day in EDA and modeling phase, we need to extract them from time column.
df.head()
df.tail()

# we have datatimeindex, extract it to year, month, weekday, day, hour and minute
# df["year"] = df.index.year
df["quarter"] = df.index.quarter
df["month"] = df.index.month
df["weekday"] = df.index.weekday  # 0 is Monday
df["day"] = df.index.day
df["hour"] = df.index.hour
# df["minute"] = df.index.minute

#  based on column hour, add a new column to indicate the different period (Night, Morning, Afternoon and Evening ) of a day.

#     Night : 22:00 - 23:59 / 00:00 - 03:59
#     Morning : 04:00 - 11:59
#     Afternoon : 12:00 - 16:59
#     Evening : 17:00 - 21:59

# We can create timing variable based on hour variable.
df["hour"].unique()
df["timing"] = "Night"
df.loc[(df["hour"] >= 4) & (df["hour"] < 12), "timing"] = "Morning"
df.loc[(df["hour"] >= 12) & (df["hour"] < 17), "timing"] = "Afternoon"
df.loc[(df["hour"] >= 17) & (df["hour"] < 22), "timing"] = "Evening"

# convert timing column to categorical type
cat_timing = CategoricalDtype(
    ["Morning", "Afternoon", "Evening", "Night"], ordered=True
)
df["timing"] = df["timing"].astype(cat_timing)

df.head()


# ----------------------------------------------------------------
# Time series analysis based on Month, Weekday, Hour, Timing
# ----------------------------------------------------------------
# Function to iterate every application to visualize the averge consuption group by month/weekday/Hour/Timing column
def plot_average_consumption(data, app_columns, groupby_column):
    for app in app_columns:
        # Group by and calculate the average consumption
        avg_df = data.groupby(groupby_column)[app].mean().reset_index()

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(avg_df[groupby_column], avg_df[app], marker="o", linestyle="-")
        plt.title(f"Average Consumption for {app}")
        plt.xlabel(f"{groupby_column}")
        plt.ylabel("Average Consumption")
        if groupby_column == "month":
            plt.xticks(
                np.arange(1, 13),
                labels=[
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
            )
        elif groupby_column == "weekday":
            plt.xticks(
                np.arange(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            )
        elif groupby_column == "hour":
            plt.xticks(np.arange(24))
        elif groupby_column == "timing":
            plt.xticks(
                np.arange(4), labels=["Morning", "Afternoon", "Evening", "Night"]
            )
        elif groupby_column == "quarter":
            plt.xticks(np.arange(1, 5))
        # plt.grid(True)
        plt.tight_layout()
        plt.show()


plot_average_consumption(df, app_columns, "month")
# Fridge and Wine cellar have higher consumption in summer
plot_average_consumption(df, app_columns, "weekday")
# There is not a weekly trend in application consumption
plot_average_consumption(df, app_columns, "hour")
plot_average_consumption(df, app_columns, "timing")


df.head()
df.info()
df.columns


# the house energy consumption is highly correlated with Furnace application
# Fridge and Wine cellar and Furnace are affected by weather information
# August has the highest energy consumption
# Furnace and living room have higher consumption in winter
# Fridge and Wine cellar have higher consumption in summer
# There is not a weekly trend in application consumption
# Home office has higher consumption in the night
# House energy consumption peaks during Evening and Night

# for Microwave, we see that a usual pattern around 11am-1pm and 16pm-18pm. However, at late night there is a weird usage!


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df.to_pickle("../../data/interim/02_add_time_features.pkl")
