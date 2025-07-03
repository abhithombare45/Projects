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
# 1293 

df.location = df.location.apply(lambda x: 'other' if x in location_stat_less_than_10 else x)
len(df.location.unique())
# 242


df[df.total_sqft/df.BHK<300]

df.shape
df = df[~(df.total_sqft/df.BHK<300)]

df.price_per_sqft.describe()

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out   

df3 = df
df = remove_pps_outliers(df)


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color= 'green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area") 
    plt.ylabel("Price")
    plt. title (location)
    plt.legend()

plot_scatter_chart(df,"Rajaji Nagar")
plot_scatter_chart(df,"Hebbal")



def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df4 = df

df = remove_bhk_outliers(df)
df 


plot_scatter_chart(df, 'Hebbal')

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet" ) 
plt.ylabel ("Count")


plt.hist(df.bath, rwidth=0.8)
plt.xlabel("Number of Bathroms")
plt.ylabel("Count")


df[df.bath>df.BHK+2]
df = df[df.bath < df.BHK+2] #bath should be smaller that (bhk+2)

 df5 = df 

df = df.drop(['size', 'price_per_sqft'], axis = 'columns')
df.head()

dummies = pd.get_dummies(df.location, dtype = int)
df6 = df
dummies.head(50)

df = pd.concat([df,dummies.drop('other', axis='columns')], axis='columns')

df = df.drop('location', axis='columns')
df.shape

# now lets x be independent variable as we are predicting prices of House in Banglore.
x = df.drop('price', axis = 'columns')
x.head()

# y is nothing but Prices of houses in Banglore, Fro Training Dateset.
# which we are predicting

y = df.price
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier





