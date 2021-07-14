import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv('FuelConsumptionCo2.csv')
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head())
#viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
#print(viz.head())
#viz.hist()
#plt.show()


'''plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()'''

#Step1: Create data test
#Create mask (mặt nạ)
msk = np.random.rand(len(df))<0.8

train=cdf[msk]
print("Size of train data: ",len(train))
test=cdf[~msk]
print("Size of test data: ",len(test))

#Step2: Modeling
from sklearn import linear_model
regr = linear_model.LinearRegression()
print("Shape before convert: ",train[['ENGINESIZE']].shape,type(train[['ENGINESIZE']]))
train_x = np.asanyarray(train[['ENGINESIZE']])
print("Shape after convert: ",train_x.shape,type(train_x))
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x,train_y)

# The coefficients y_predict = theta_0 + theta_1 * x
print ('Coefficients: ', regr.coef_) #theta_0
print ('Intercept: ',regr.intercept_)#theta_1

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color="blue")
plt.plot(train_x,regr.intercept_[0]+regr.coef_[0][0]*train_x,'-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Evaluation
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_predict = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_predict - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_predict - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_predict))