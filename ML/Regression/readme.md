# sumarize
## 1.linear regression
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;y=a&plus;b*x}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;y=a&plus;b*x}" title="{\color{White} y=a+b*x}" /></a> <br>
- $a: Coefficients$ <br>
- $b: Intercept$ <br>

B1: Create train and test data: 
```python
msk = np.random.rand(len(df))<0.8 #create mask
train=cdf[msk]
test=cdf[~msk]
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
```
B2: Modeling
```python
regr = linear_model.LinearRegression()
regr.fit (train_x,train_y)
print ('Coefficients: ', regr.coef_) #theta_0
print ('Intercept: ',regr.intercept_)#theta_1

test_y_predict = regr.predict(test_x)
```
B3: Evaluate in test set
## 2.Multiple linear regression
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;y=a&plus;b*x&plus;c*y&plus;d*z}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;y=a&plus;b*x&plus;c*y&plus;d*z}" title="{\color{White} y=a+b*x+c*y+d*z}" /></a> <br>
- $a: Coefficients$ <br>
- $[b,c,d]: Intercept$ <br>

## 3.Nonlinear regression
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;log(x),e^{x},..}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;log(x),e^{x},..}" title="{\color{White} log(x),e^{x},..}" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;Ex:&space;y=\frac{1}{1&plus;e^{-a*(x-b)}}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;Ex:&space;y=\frac{1}{1&plus;e^{-a*(x-b)}}}" title="{\color{White} Ex: y=\frac{1}{1+e^{-a*(x-b)}}}" /></a> <br>
- B1: create train and test
```python
....
#Normalization data
x_data_normal = x_data/max(x_data)
y_data_normal = y_data/max(y_data)
```
- B2: create curve_fit
```python
def sigmoid(x,a,b):
    y=1/(1+np.exp(-a*(x-b)))
    return y
from scipy.optimize import curve_fit
popt,pcov = curve_fit(sigmoid,x_data_normal,y_data_normal)
y_predict = sigmoid(x,*popt)
```
- B3: Evaluation test data

## 4. Polynomial regression
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;y=a&plus;b*x&plus;c*x^2&plus;d*x^3&plus;...}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;y=a&plus;b*x&plus;c*x^2&plus;d*x^3&plus;...}" title="{\color{White} y=a+b*x+c*x^2+d*x^3+...}" /></a><br>

- B1: Create data train and test
- B2: Preprocessing data
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x) #convert x to 0,x,x^2,x^3
```
- B3: Modeling with data_poly
## 5. Some metrics
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;MAE=\frac{1}{n}\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;MAE=\frac{1}{n}\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|}" title="{\color{White} MAE=\frac{1}{n}\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|}" /></a> <br>

<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}" title="{\color{White} MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}" /></a><br>

<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}" title="{\color{White} MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}" /></a> <br>

<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;RAE=\frac{\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|}{\sum_{i=1}^{n}|y_{i}-\overline{y}_{i}|}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;RAE=\frac{\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|}{\sum_{i=1}^{n}|y_{i}-\overline{y}_{i}|}}" title="{\color{White} RAE=\frac{\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|}{\sum_{i=1}^{n}|y_{i}-\overline{y}_{i}|}}" /></a> <br>

<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;RSE=\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{\sum_{i=1}^{n}(y_{i}-\overline{y}_{i})^2}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;RSE=\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{\sum_{i=1}^{n}(y_{i}-\overline{y}_{i})^2}}" title="{\color{White} RSE=\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{\sum_{i=1}^{n}(y_{i}-\overline{y}_{i})^2}}" /></a><br>

<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;Rsquared:&space;R^2&space;=&space;1-&space;RSE}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;Rsquared:&space;R^2&space;=&space;1-&space;RSE}" title="{\color{White} Rsquared: R^2 = 1- RSE}" /></a><br>


