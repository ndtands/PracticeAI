import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("FuelConsumptionCo2.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

mks = np.random.rand(len(df)) <0.8
train = cdf[mks]
test = cdf[~mks]
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
train_y= np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
test_y= np.asanyarray(test[['CO2EMISSIONS']])
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(train_x,train_y)

# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

y_predict = regr.predict(test_x)
from sklearn.metrics import r2_score

print("Residual sum of squares: %.2f"% np.mean((y_predict - test_y) ** 2))
print('Variance score: %.2f' % regr.score(test_x, test_y))
print("R2-score: %.2f" % r2_score(test_y , y_predict))

### R2= 1-RSS/TSS
####Variance score = 1-var(y-y_predict)/var(y)
print("%.2f" %(1-np.var(test_y-y_predict)/np.var(test_y)))