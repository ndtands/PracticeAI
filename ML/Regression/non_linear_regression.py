import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Ex: y=2x+3
'''x   = np.arange(-5,5,0.1)
y   = 2*x+3
y_noise = 2*np.random.normal(size=x.size) #2 standar
y_data = y+y_noise
plt.plot(x, y_data,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()'''
## log exp,...

df = pd.read_csv("china_gdp.csv")
x_data ,y_data = df['Year'].values,df["Value"].values
'''plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()'''

#choice model is sigmoid
def sigmoid(x,Beta_1,Beta_2):
    y=1/(1+np.exp(-Beta_1*(x-Beta_2)))
    return y
#Normalize our data
x_data_normal = x_data/max(x_data)
y_data_normal = y_data/max(y_data)
'''plt.plot(x_data_normal, y_data_normal, 'ro')
plt.ylabel('GDP_normalize')
plt.xlabel('Year_normalize')
plt.show()
'''
from scipy.optimize import curve_fit
popt,pcov = curve_fit(sigmoid,x_data_normal,y_data_normal)
print(" beta1 =%f ,beta2=%f"%(popt[0],popt[1]))
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y_predict = sigmoid(x,*popt)
plt.plot(x_data_normal,y_data_normal,'ro',label="data")
plt.plot(x,y_predict, linewidth=3.0, label='fit')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


##########Practice avaluate
msk =np.random.rand(len(df))<0.8
train_x = x_data_normal[msk]
train_y = y_data_normal[msk]
test_x = x_data_normal[~msk]
test_y = y_data_normal[~msk]
#build the model using train set
popt,pcov = curve_fit(sigmoid, train_x, train_y)
y_predict = sigmoid(test_x,*popt)

print("Mean absolute error (MAE): %.6f"%np.mean(np.absolute(y_predict - test_y)))
print("Resiual sum of squares (MSE): %.6f"%np.mean((y_predict-test_y)**2))
from sklearn.metrics import r2_score
print("R2-score: %.6f" % r2_score(y_predict , test_y) )