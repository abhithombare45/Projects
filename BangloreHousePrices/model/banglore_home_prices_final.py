import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %matplotlib inline
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20, 10)

df = pd.read_csv("Bengaluru_House_Data.csv.xls")

df.head()

df.shape

df.groupby("area_type")["area_type"].agg("count")

df1 = df
df.head()

df = df.drop(["area_type", "availability", "society", "balcony"], axis="columns")

df.head()
df.isnull().head(50)
df.isnull().sum()

df = df.dropna()

df["size"].unique()

df["BHK"] = df["size"].apply(lambda x: int(x.split(" ")[0]))

df["BHK"].unique()

df[df.BHK > 15].head(50)

df.total_sqft.unique()
# df[df.total_sqft== ...]


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df[~df["total_sqft"].apply(is_float)].head()

df2 = df


def convert_sqft_to_num(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


convert_sqft_to_num("45 - 93")
convert_sqft_to_num("45 - ")  # emitting error
convert_sqft_to_num("45.93sqft area")  # do nothing

df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
df.loc[30]

df['price_per_sqft'] = df['price']*100000/df['total_sqft']

df.head()

#   df.location.unique().  # list of unique locations
len(df.location.unique())

df.location = df.location.apply(lambda x: x.strip())

location_stat = df.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stat.head(50)

    location_stat[location_stat<=10]
len(location_stat[location_stat<=10])


location_stat_less_than_10 = location_stat[location_stat<=10]

len(df.location.unique())
#1293 


