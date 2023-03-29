# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
import statsmodels as sm

df = pd.read_excel('weekly_prepared_26_11_2017.xlsx', sheet_name='JTI_weekly_prepared_26_11_2017')

# set week + id as the time series index
df = df.assign(weekid=df['week']*1000000000000+df['id'])
df.set_index('weekid', inplace=True)
X= df[['mileage','avg_speed','drg1_100','side1_100','avg_daily_business_mileage','acc1_100','speed3_100']]
Xtempt = ['mileage','avg_speed','drg1_100','side1_100','avg_daily_business_mileage','acc1_100','speed3_100']
Y= df['crash']

# # Backward Elimination
import statsmodels.api as sm
import operator
def backward_elimination(X, Y):
    numVars = len(X[0])
    for i in range(0, (numVars-1)):
        regressor_OLS = sm.OLS(Y, X).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > 0.05:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    X = np.delete(X, j, 1)
    return X

X = X.values.reshape(-1, 7)







# test ssz
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = df.rolling(window=3).mean(timeseries)
    rolstd = df.rolling(window= 3).std(timeseries)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print(dftest)
    for key,value in dftest.items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput) 

# first difference
df['crash'] = df['crash'] - df['crash'].shift(1)
df = df.dropna()
test_stationarity(df['crash'])

# second difference
df['crash'] = df['crash'] - df['crash'].shift(1)
df = df.dropna()
test_stationarity(df['crash'])

# third difference
df['crash'] = df['crash'] - df['crash'].shift(1)
df = df.dropna()
test_stationarity(df['crash'])


# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Create a random forest regressor with 100 trees and set some controls
rf=RandomForestRegressor(n_estimators=100, max_features=2, max_depth=10, min_samples_split=2, min_samples_leaf=1, bootstrap=True, oob_score=False, n_jobs=1, random_state=0, verbose=0)

# Fit the model to the training data
rf.fit(X_train, Y_train)

# Predict on the test set and calculate the mean squared error
Y_pred = rf.predict(X_test)
mse = ((Y_pred - Y_test) ** 2).mean()
print('MSE before BE = ')
print(mse)

# # Backward Elimination
import statsmodels.api as sm
import operator
def backward_elimination(X, Y):
    numVars = len(X[0])
    for i in range(0, (numVars-1)):
        regressor_OLS = sm.OLS(Y, X).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > 0.05:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    X = np.delete(X, j, 1)
    return X

X_Modeled = backward_elimination(X, Y)


# Fit the model to the training data
rf.fit(X_train, Y_train)

# Predict on the test set and calculate the mean squared error
Y_pred = rf.predict(X_test)
mse2 = ((Y_pred - Y_test) ** 2).mean()

# Print the mean squared error 
print('MSE after BE = ')
print(mse2)

# Print the coefficient of determination (OOB) of the prediction
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)
RFC.fit(X_train,Y_train)
print(RFC.oob_score_)


# Print the feature ranking named by the random forest regressor
print("Feature ranking:")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, Xtempt[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
         color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Plot the predicted values against the actual values
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()


# Print the average predicted value for crashers and non-crashers
sum_crash=0
sum_not_crash=0
count_crash=0
count_not_crash=0
for i in range(len(Y_test.values)) :
    if Y_test.values[i] == 0 :
        sum_not_crash+=Y_pred[i]
        count_not_crash+=1
    else:
        sum_crash+=Y_pred[i]
        count_crash+=1
print('Avg for crashers: ', sum_crash/count_crash)
print('Avg for non-crashers: ', sum_not_crash/count_not_crash)

# # Visualising the Random Forest Regression results
 
# # arrange for creating a range of values from min value of x to max
# # value of x with a difference of 0.01 between two consecutive values
# X_grid = np.arrange(min(X['mileage']), max(X['mileage']), 0.01)
 
# # reshape for reshaping the data into a len(X_grid)*1 array,
# # i.e. to make a column out of the X_grid value                 
# X_grid = X_grid.reshape((len(X_grid), 1))
 
# # Scatter plot for original data
# plt.scatter(x, y, color = 'blue') 
    
# # plot predicted data
# plt.plot(X_grid, regressor.predict(X_grid),
#          color = 'green')
# plt.title('Random Forest Regression')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()