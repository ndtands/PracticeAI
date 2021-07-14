# 1.KNN - K Nearest neighbors (non-parametric)
## 1.1 Algorithm
- Pick a value for k
- Calculate the distance from A point to all point
- Select the k-distance in the train dataset are "nearest" to the A point
- Predict the response of the A point using popular respone value from the KNN

## 1.2 Evaluation
- Jaccard index <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;J(y,\hat{y})&space;=&space;\frac{|y\cap&space;\hat{y}|}{|y|&plus;|\hat{y}|-|y\cap&space;\hat{y}|}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;J(y,\hat{y})&space;=&space;\frac{|y\cap&space;\hat{y}|}{|y|&plus;|\hat{y}|-|y\cap&space;\hat{y}|}}" title="{\color{White} J(y,\hat{y}) = \frac{|y\cap \hat{y}|}{|y|+|\hat{y}|-|y\cap \hat{y}|}}" /></a>
- F1-score <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;F1=\frac{2.Pre.Re}{Pre&plus;Re}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;F1=\frac{2.Pre.Re}{Pre&plus;Re}}" title="{\color{White} F1=\frac{2.Pre.Re}{Pre+Re}}" /></a>
- Log loss <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;logloss=\frac{-1}{n}\sum[&space;y*log(\hat{y})&plus;(1-y)*log(1-\hat{y})]&space;}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;logloss=\frac{-1}{n}\sum[&space;y*log(\hat{y})&plus;(1-y)*log(1-\hat{y})]&space;}" title="{\color{White} logloss=\frac{-1}{n}\sum[ y*log(\hat{y})+(1-y)*log(1-\hat{y})] }" /></a>

## 1.3 code python
- B1: Normaliza data
```python
y = df['custcat'] #=> series
y = df['custcat'].values #=> ndarray
from sklearn import preprocessing
....
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
```
- B2: Load test set and train set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
```
- B3: Classification and predict
```python
from sklearn.neighbors import KNeighborsClassifier
k=4
neigh4 =KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
y_predict = neigh4.predict(X_test)
```
- B4: Accuracy evaluation with jaccard_score
```python
from sklearn import metrics
print("Train set Accuracy with k=4: ", metrics.accuracy_score(y_train, neigh4.predict(X_train)))
print("Test set Accuracy with k=4: ", metrics.accuracy_score(y_test, y_predict))
```
- B5: Find the best k
```python
ks=20
mean_acc = np.zeros(ks-1)
std_acc = np.zeros(ks-1)
for n in range(1,ks):
     neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    y_predict = neigh.predict(X_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,y_predict)
    # std= std_true/sqrt(n)
    std_acc[n-1]=np.std(y_predict==y_test)/np.sqrt(y_predict.shape[0])

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
```
- B6: Visulization
```python
#Plot model accuracy for Different number of Neighbors
plt.plot(range(1,ks),mean_acc,'g')
plt.fill_between(range(1,ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
```
## 1.4. Advantages of KNN
- The computational complexity of the training process is zero
- Predict the outcome of new data is simple
- There is no need to make any assumptions about the distribution of classes.

## 1.5. Disadvatages of KNN
- KNN is very sensitive to noise when K is small
- KNN is an algorithm where all calculations are in testing stage. When K is large and large dimensional database. So, the computational complexity will increase. In other hand, storing all data in memory also affects the performance of KNN.
# 2.Logistic Regression
## 2.1. Algorithm
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;\hat{y}=\sigma{(\theta^{T}.X)}=P(y=1|x)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;\hat{y}=\sigma{(\theta^{T}.X)}=P(y=1|x)}" title="{\color{White} \hat{y}=\sigma{(\theta^{T}.X)}=P(y=1|x)}" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;\sigma{(\theta^{T}.X)}=\frac{1}{1&plus;e^{-\theta^{T}.X}}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;\sigma{(\theta^{T}.X)}=\frac{1}{1&plus;e^{-\theta^{T}.X}}}" title="{\color{White} \sigma{(\theta^{T}.X)}=\frac{1}{1+e^{-\theta^{T}.X}}}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;P(y=0|X)=1-\sigma{(\theta^{T}.X)}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;P(y=0|X)=1-\sigma{(\theta^{T}.X)}}" title="{\color{White} P(y=0|X)=1-\sigma{(\theta^{T}.X)}}" /></a> <br>
- Init the Parameter random <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;\theta=[\theta_{0},\theta_{1},\theta_{2},...]}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;\theta=[\theta_{0},\theta_{1},\theta_{2},...]}" title="{\color{White} \theta=[\theta_{0},\theta_{1},\theta_{2},...]}" /></a>
- Feed the cost function with train set and caculate err <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;Cost(\hat{y},y)=\frac{1}{2}(\sigma(\theta^{T}.X)-y)^2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;Cost(\hat{y},y)=\frac{1}{2}(\sigma(\theta^{T}.X)-y)^2}" title="{\color{White} Cost(\hat{y},y)=\frac{1}{2}(\sigma(\theta^{T}.X)-y)^2}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;=>&space;J(\theta)=\frac{1}{m}\sum^{m}_{i=1}(cost(\hat{y},y))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;=>&space;J(\theta)=\frac{1}{m}\sum^{m}_{i=1}(cost(\hat{y},y))}" title="{\color{White} => J(\theta)=\frac{1}{m}\sum^{m}_{i=1}(cost(\hat{y},y))}" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;y=1&space;|&space;y=0&space;=>&space;J(\theta)=\frac{1}{m}\sum^{m}_{i=1}[y_{i}.log(\hat{y}_{i})&plus;(1-y_{i}).log(1-\hat{y}_{i})]}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;y=1&space;|&space;y=0&space;=>&space;J(\theta)=\frac{1}{m}\sum^{m}_{i=1}[y_{i}.log(\hat{y}_{i})&plus;(1-y_{i}).log(1-\hat{y}_{i})]}" title="{\color{White} y=1 | y=0 => J(\theta)=\frac{1}{m}\sum^{m}_{i=1}[y_{i}.log(\hat{y}_{i})+(1-y_{i}).log(1-\hat{y}_{i})]}" /></a>
- Caculate the gradient of cost funtion <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;\nabla&space;J=\begin{bmatrix}&space;\frac{\partial&space;J}{\partial&space;\theta_{1}}&space;&&space;\frac{\partial&space;J}{\partial&space;\theta_{2}}&space;&...&space;&&space;\frac{\partial&space;J}{\partial&space;\theta_{n}}&space;\end{bmatrix}^{T}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;\nabla&space;J=\begin{bmatrix}&space;\frac{\partial&space;J}{\partial&space;\theta_{1}}&space;&&space;\frac{\partial&space;J}{\partial&space;\theta_{2}}&space;&...&space;&&space;\frac{\partial&space;J}{\partial&space;\theta_{n}}&space;\end{bmatrix}^{T}}" title="{\color{White} \nabla J=\begin{bmatrix} \frac{\partial J}{\partial \theta_{1}} & \frac{\partial J}{\partial \theta_{2}} &... & \frac{\partial J}{\partial \theta_{n}} \end{bmatrix}^{T}}" /></a>
- Update weight <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;\theta_{new}=\theta_{old}-\eta.\nabla&space;j&space;}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;\theta_{new}=\theta_{old}-\eta.\nabla&space;j&space;}" title="{\color{White} \theta_{new}=\theta_{old}-\eta.\nabla j }" /></a>
- Go to step 2 when J is small 
- Predict


## 2.2. Code python
- Normalize data
    - MaxminScaler <br>
    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;y&space;=&space;\frac{x-min}{max&space;-min}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;y&space;=&space;\frac{x-min}{max&space;-min}}" title="{\color{White} y = \frac{x-min}{max -min}}" /></a>
    - StandardScaler <br>
    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;y&space;=&space;\frac{x_{i}-mean(x)}{std(x)}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;y&space;=&space;\frac{x_{i}-mean(x)}{std(x)}}" title="{\color{White} y = \frac{x_{i}-mean(x)}{std(x)}}" /></a>
```python
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
```
- Load dataset for testing and training
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
```
- Modeling
```python
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
y_predict = LR.predict(X_test)
'''
predict_proba returns estimates for all classes, ordered by the label of classes. 
So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
'''
y_predict_prob = LR.predict_proba(X_test)
```
- Evaluation with jaccard_score
```python
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, y_predict,pos_label=0))
```
- Evaluation with confusion matrix
```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_predict,labels=[1,0]))
print(classification_report(y_test,y_predict))
```
- Evaluation with log loss
```python
from sklearn.metrics import log_loss
print(log_loss(y_test, y_predict_prob))
```

# 3. Decision trees
## 3.1. Algorithm
- Choose an atribute from your dataset
- Caculate the signification of attribute in spliting of data
- Split data based on the valuse of the best attribute
- Go to step 1 <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{White}&space;Entropy&space;=&space;-P(A).log(P(A))-P(B).log(P(B))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;{\color{White}&space;Entropy&space;=&space;-P(A).log(P(A))-P(B).log(P(B))}" title="{\color{White} Entropy = -P(A).log(P(A))-P(B).log(P(B))}" /></a> <br>
$Information.Gain= (Entropy.before.split) - (weight.Entropy.after.split)$
- Using jaccard_score to evaluate model

## 3.2. Code with python
- preprocessing data
```python
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])  #0,1

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2]) #0,1,2

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) #0,1
y = my_data["Drug"]
```
- Split data train and set
```python
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))
```
- Modeling and predict
```python
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)
#predict
predTree = drugTree.predict(X_testset) #=>ndarray
```
- Evaluation with jaccard_score 
```python
#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
```

# 4. SVM - support vector machine
- SVM algorithm can perform really well with both linearly separable and non-linearly separable datasets
- if you want using the sigmoid function, you must normalization
## 4.1 Code python
```python
from sklearn import svm
#clf = svm.SVC(kernel='rbf')
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
print("w =", clf.coef_,"\n","b =", clf.intercept_)
yhat = clf.predict(X_test)
#Evaluate
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat,pos_label=2))
```
## 4.2. Advantages of Support Vector Machine Algorithm
- Accuracy
- Works very well with limited datasets
- Kernel SVM contains a non-linear transformation function to convert the complicated non-linearly separable data into linearly separable data.

## 4.3. Disadvantages of Support Vector Machine Algorithm
- Does not work well with larger datasets
- Sometimes, training time with SVMs can be high