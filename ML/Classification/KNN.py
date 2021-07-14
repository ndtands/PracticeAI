import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('teleCust1000t.csv')
#print(df.head())
#print(df['custcat'].value_counts())
#281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

#visualize icome
'''viz =df[['income']]
viz.hist(bins=50)
plt.show()'''
#print(df.columns)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
#print(X[0:5])
y = df['custcat'] #=> series
y = df['custcat'].values #=> ndarray
#print(y[0:5])

#Normalize data with zero mean and unit variance
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
#Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Classification
from sklearn.neighbors import KNeighborsClassifier
k=4
neigh =KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

#Predict
y_predict = neigh.predict(X_test)

#Accurary evaluation
from sklearn import metrics
print("Train set Accuracy with k=4: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy with k=4: ", metrics.accuracy_score(y_test, y_predict))

#####################################Practice with k=6###################
k=6
neigh6 = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
print(neigh6)
y_predict6 = neigh6.predict(X_test)
print(y_predict6.shape)
print("Train set Accuracy with k=6: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy with k=6: ", metrics.accuracy_score(y_test, y_predict6))


#### choice the best k#####
ks=20
mean_acc = np.zeros(ks-1)
std_acc = np.zeros(ks-1)
for n in range(1,ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    y_predict = neigh.predict(X_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,y_predict)
    std_acc[n-1]=np.std(y_predict==y_test)/np.sqrt(y_predict.shape[0])

#Plot model accuracy for Different number of Neighbors
plt.plot(range(1,ks),mean_acc,'g')
plt.fill_between(range(1,ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)