import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv("FuelConsumptionCo2.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#Create train and test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x) #convert x to 0,x,x^2,x^3


clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

'''plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()'''


#### Evaluation
from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
y_predict =clf.predict(test_x_poly)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predict - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_predict - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,y_predict ))


######################Practice with degree = 4 ####################
#create train and test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly4 = PolynomialFeatures(degree=4)
train_x_poly4=poly4.fit_transform(train_x)

clf4 = linear_model.LinearRegression().fit(train_x_poly4,train_y)

# The coefficients
print ('Coefficients: ', clf4.coef_)
print ('Intercept: ',clf4.intercept_)


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf4.intercept_[0]+ clf4.coef_[0][1]*XX + clf4.coef_[0][2]*np.power(XX, 2) + clf4.coef_[0][3]*np.power(XX, 3)+ clf4.coef_[0][4]*np.power(XX, 4)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
test_x_poly4 = poly4.transform(test_x)
y_predict = clf4.predict(test_x_poly4)
print("MAE = %.4f"%np.mean(np.absolute(test_y-y_predict)))
print("MSE = %.2f"%np.mean((y_predict-test_y)**2))
print("R2-score = %.2f"%r2_score(test_y,y_predict))
print('Variance score: %.2f' % clf4.score(test_x_poly4, test_y))