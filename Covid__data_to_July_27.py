# -*- coding: utf-8 -*-
"""
Created on Tues 28 July 14:57:58 2020

@author: Owner# Tanya Reeves

Aims: 
1. Track the number of confirmed, recovered and deaths in every country globally, 
   work out top 15 as of 27 July 2020, do basic visualisations
2. Look at correlations between these variables (lat and long preserved as a sanity check)
3. Linear regression model 1: confirmed:fatalities, scatterplot and predicted values
4. Linear regression model 2: confirmed:recovered, scatterplot and predicted values
5. Evaluate both these models (R2, MAE, MSE), suggesting future non-linear models

"""
#==================================================================================

# IMPORT LIBRARIES

#==================================================================================

# manipulating data
import numpy as np
import pandas as pd
# visualisations
import matplotlib.pyplot as plt
import seaborn as sns

#===============================================================================

# WRANGLING DATA

#===============================================================================

df = pd.read_csv("covid_19_clean_complete.csv")

df.head(3)

df.isnull().sum()
"""
Province/State    34404
Country/Region        0
Lat                   0
Long                  0
Date                  0
Confirmed             0
Deaths                0
Recovered             0
Active                0
WHO Region            0
dtype: int64
"""
#---------------
# RENAME COLUMNS
# - use Dictionary to rename columns, old column name is key, new column name is value
# - dictionary {"old_name": "new_name"}
#---------------
df.columns
"""
Index(['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'Confirmed',
       'Deaths', 'Recovered', 'Active', 'WHO Region'],
      dtype='object')
"""
df = df.rename(
        columns = {'Province/State':'subregion'
                   ,'Country/Region':'country'
                   ,'Lat':'lat'
                   ,'Long':'long'
                   ,'Date':'date'
                   ,'Confirmed':'confirmed'
                   ,'Deaths':'deaths'
                   ,'Recovered':'recovered'
                   ,'Active':'active'
                   ,'WHO Region': 'who_region'
                   }
        )

df.columns
"""
Index(['subregion', 'country', 'lat', 'long', 'date', 'confirmed', 'deaths',
       'recovered', 'active', 'who_region'],
      dtype='object')
"""
# rearrange as we want all dates in on column
(df.filter(['date'])
)
"""
             date
0      2020-01-22
1      2020-01-22
2      2020-01-22
3      2020-01-22
4      2020-01-22
          ...
49063  2020-07-27
49064  2020-07-27
49065  2020-07-27
49066  2020-07-27
49067  2020-07-27
"""

df.shape
# (49068, 10)

#=====================
# SORT & REARANGE DATA
#=====================


df = (df.filter(['country', 'subregion', 'date', 'lat', 'long', 'confirmed', 'deaths', 'recovered', 'active', 'who_region'])
               .sort_values(['country','subregion','lat','long','date', 'confirmed', 'deaths', 'recovered', 'active', 'who_region'])
               
               )
df.columns

#-------------------------------------------------

# SET INDEX
# - filter according to country

#-------------------------------------------------
df.set_index('country', inplace = True)
pd.set_option('display.max_rows', 200)
(df
    .reset_index()
    .filter(['country'])
    .drop_duplicates()
    .head(n = 200)
)

print(df)
pd.reset_option('display.max_rows')

#--------------------------------------------------
# GET SUMMARY DATE FOR: confirmed, deaths, recovered
#--------------------------------------------------
(df
 .filter(['confirmed','deaths','recovered'])
 .describe()
)
"""
          confirmed         deaths     recovered
count  4.906800e+04   49068.000000  4.906800e+04
mean   1.688490e+04     884.179160  7.915713e+03
std    1.273002e+05    6313.584411  5.480092e+04
min    0.000000e+00       0.000000  0.000000e+00
25%    4.000000e+00       0.000000  0.000000e+00
50%    1.680000e+02       2.000000  2.900000e+01
75%    1.518250e+03      30.000000  6.660000e+02
max    4.290259e+06  148011.000000  1.846641e+06
"""

# ======================================================================

# CREATE BASIC VISUALISATIONS

# ======================================================================

# .iloc means 'index location' and slices subsections of the data
# first the good news - top 15 countries in terms of recoveries
recovered_by_country_top15 = (df
                        #.query('date == datetime.date()')
                        .filter(['country','recovered'])
                        .groupby('country')
                        .agg('sum')
                        .sort_values('recovered', ascending = False)
                        .reset_index()
                        .iloc[0:15,:]
                        )

sns.barplot(data = recovered_by_country_top15
            ,y = 'country'
            ,x = 'recovered'            
        )

# now the number of confirmed top 15 (sliced by we are going to use .iloc)
confirmed_by_country_top15 = (df
                        #.query('date == datetime.date()')
                        .filter(['country','confirmed'])
                        .groupby('country')
                        .agg('sum')
                        .sort_values('confirmed', ascending = False)
                        .reset_index()
                        .iloc[0:15,:]
                        )

sns.barplot(data = confirmed_by_country_top15
            ,y = 'country'
            ,x = 'confirmed'
        )

#=================================================================================

# CORRELATIONS

#================================================================================

# we see if there are any obvious correlations
# a correlation matrix reveals a high correlation between cases and fatalities (0.93)
# (to recap, a correlation of 1 is a perfect correlation, 0 is no correlation, the nearer 1, the higher the correlation)
# the diagonals show the distributions of the variables. Not a perfect Normal distribution

corr_matrix = covid_data_2020JULY27.corr()
corr_matrix["deaths"].sort_values(ascending=False)
"""
deaths       1.000000
confirmed    0.925647
recovered    0.791999
lat          0.128400
long        -0.050017
Name: deaths, dtype: float64
"""

corr_matrix = covid_data_2020JULY27.corr()
corr_matrix["confirmed"].sort_values(ascending=False)
# confirmed very highly correlated with deaths, second highest with recovered
"""
confirmed    1.000000
deaths       0.925647
recovered    0.895395
lat          0.076977
long        -0.012772
Name: confirmed, dtype: float64

"""
# plotting a scatter matrix looks at all combinations
# as a sub-aim, thinking if there is a pattern with longitude and latitude

from pandas.plotting import scatter_matrix
attributes = ["deaths", "recovered", "confirmed", "lat", "long"]
scatter_matrix(covid_data_2020JULY27[attributes], figsize=(12,8))

# ========================================================================

# research aim 1: confirmed cases:mortality ratio
# research aim 2: confirmed cases:recovered ratio

# =========================================================================

# -------------------------------------------------------------------------------

# BASIC SCATTERPLOT

# -------------------------------------------------------------------------------
# if there is a good correlation, there will be a definite pattern
# if there is a pattern, it suggests a good rationale for using one variable to predict another

covid_data_2020JULY27 = (df
                        #.query("date == datetime.date(2020, 4, 9)")
                        .filter(['country', 'lat', 'long', 'confirmed','deaths','recovered'])
                        .groupby('country')
                        .agg('sum')
                        .sort_values('confirmed', ascending = False)
                        .reset_index()
)

print(covid_data_2020JULY27)

# ratio of confirmed: deaths
sns.scatterplot(data = covid_data_2020JULY27
                ,x = 'confirmed'
                ,y = 'deaths'
                )

# ratio of confirmed:recovered is a more obvious line of best fit (better scaling)
sns.scatterplot(data = covid_data_2020JULY27
                ,x = 'confirmed'
                ,y = 'recovered'
                )

# ===========================================================================

# SIMPLE LINEAR REGRESSION 1 (confirmed: fatalities)
# Can we predict fatalities from confirmed case numbers?
# To do this, we use a simple linear regression and evaluate its efficiency

#=============================================================================

# check the shape of the dataframe
covid_data_2020JULY27.shape
# 187 x 6 ie 187 countries
covid_data_2020JULY27.columns
# Index(['country', 'lat', 'long', 'confirmed', 'deaths', 'recovered'], dtype='object')

covid_data_2020JULY27.isnull().sum()
# check no missing data
"""
Out[43]: 
country      0
lat          0
long         0
confirmed    0
deaths       0
recovered    0
dtype: int64
"""

# as it is a pandas dataframe, we need to tell it to extract the values
X = covid_data_2020JULY27[['confirmed']].values
y = covid_data_2020JULY27[['deaths']].values

# 80/20 split in training/test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# check the size of the training and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# (149, 1) (38, 1) (149, 1) (38, 1)
# this means X_train y_train have 149 countries, X_text and y_test have 38

# now import the package to do the linear regression
from sklearn.linear_model import LinearRegression

# run the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print("Y intercept: ", linear_reg.intercept_)
print("Coefficient b1 (slope): ", linear_reg.coef_)

"""
Y intercept:  [-15336.76536398]
Coefficient b1 (slope):  [[0.04754507]]
"""
# confirmed:deaths
# y = -15336.77 + 0.05x
# obviously here y-intercept in meaningless as we cannot have minus confirmed cases
# small gradient here means for every confirmed case, hard to make a call on fatalities
# ie the virus is random, and hard to predict (we already knew that!)

y_pred = linear_reg.predict(X_test)
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, color = 'red')
plt.show()

df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1.head(15)

"""

     Actual     Predicted
0      2181 -6.969832e+03
1    140031  9.701851e+04
2         0 -1.452855e+04
3     25664  3.262770e+04
4      2696 -1.050576e+04
5     16373  3.521226e+03
6    699566  4.295211e+05
7      3520 -7.415424e+03
8      1346 -1.442918e+04
9   3048524  9.931382e+05
10  3997775  1.256427e+06
11     5250 -9.318035e+03
12     1078 -1.425930e+04
13     4717  1.371622e+04
14      705 -7.082846e+03

"""
# looking at the plot, confirmed and recovered would be better to examine together
df2 = covid_data_2020JULY27.head(15)
df2.plot(kind = 'bar', figsize = (10,8))
plt.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'green')
plt.grid(which = 'minor', linestyle = '-', linewidth = '0.5', color = 'green')

# to quantify the accuracy, R2 score is 0.528 which is a low score, ie poor model
# (to recap, R2 is measure of how close the data are to the regression line)
# (1 indicates that the model explains all the variability of the response)

from sklearn import metrics
from sklearn.metrics import r2_score
print("R2 score: ", r2_score(y_test, y_pred))
# R2 score:  0.5279529959166951
# poor R2 score

# Other metrics are Mean Absolute Error (MAE), Mean Squared Error (MSE)
# and Root Mean Squared Error (RMSE). The smaller the number the better the model's accuracy.
# RMSE is square root of MSE and is more sensitive to outliers except when outliers are exponentially rare (as in this case)
# MSE also more sensitive to cases with greatest difference
# MAE is much smaller and would suggest a reasonably good fit (but not perfect!)

print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

"""
Mean Absolute Error:  197938.18389212005
Mean Squared Error:  392112643559.5556
Root Mean Squared Error:  626188.9839014701
"""

# ===========================================================================

# SIMPLE LINEAR REGRESSION 2 confirmed:recovered
# Since the regression between confirmed cases and fatalities has high error
# we can go one step further: can we predict recoveries from confirmed case numbers?
# To do this, we use a simple linear regression and evaluate its efficiency

#=============================================================================

# now confirmed cases are predicting recoveries
X = covid_data_2020JULY27[['confirmed']].values
y = covid_data_2020JULY27[['recovered']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# check the size of the training and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# (149, 1) (38, 1) (149, 1) (38, 1)
# this means X_train y_train have 149 countries, X_text and y_test have 38

# run the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print("Y intercept: ", linear_reg.intercept_)
print("Coefficient b1 (slope): ", linear_reg.coef_)

"""
Y intercept:  [790946.60921326]
Coefficient b1 (slope):  [[0.32779499]]
"""
# confirmed:deaths
# y = 790946.6 + 0.33x
# here the regression line is more meaningful
# for every confirmed case globally, there is a 1 in 3 chance of a fatality
# however, as the metrics concur, this is not a well-fitting model
# it would suggest much more sophisticated models to investigate this phenomenon

y_pred = linear_reg.predict(X_test)
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, color = 'red')
plt.show()

df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1.head(15)

"""
     Actual     Predicted
0    149560  8.486316e+05
1   1409409  1.565569e+06
2      9902  7.965188e+05
3    544430  1.121633e+06
4     29132  8.242535e+05
5    207156  9.209612e+05
6         0  3.857977e+06
7     82508  8.455595e+05
8      8521  7.972039e+05
9   7182115  7.743782e+06
10   126217  9.558999e+06
11    85410  8.324422e+05
12     6573  7.983751e+05
13   309368  9.912497e+05
14   106302  8.478525e+05

"""

# (to recap, R2 is measure of how close the data are to the regression line)
# (1 indicates that the model explains all the variability of the response)

print("R2 score: ", r2_score(y_test, y_pred))
# R2 score:  0.43042900493300573
# very poor R2 score

# Other metrics are Mean Absolute Error (MAE), Mean Squared Error (MSE)
# and Root Mean Squared Error (RMSE). The smaller the number the better the model's accuracy.
# RMSE is square root of MSE and is more sensitive to outliers except when outliers are exponentially rare (as in this case)
# MAE is much smaller and would suggest a reasonably good fit (but not perfect!)

print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

"""
MAE larger than confirmed:fatalities. MSE huge, suggesting non-linear model much better

Mean Absolute Error:  1120706.9163624502
Mean Squared Error:  3938580753904.465
Root Mean Squared Error:  1984585.7890009354
"""
