
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

churn_df = pd.read_csv("ChurnData.csv")
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print("Shape: ",churn_df.shape)

#Data set
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

#nomarlization
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

#Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
y_predict = LR.predict(X_test)
#print(y_predict[:10])
'''
predict_proba returns estimates for all classes, ordered by the label of classes. 
So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
'''
y_predict_prob = LR.predict_proba(X_test)
#print(y_predict_prob[:10])

#Evaluation with jaccard_score
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, y_predict,pos_label=0))

#Evaluation with confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
print("confusion matrix")
print(confusion_matrix(y_test,y_predict,labels=[1,0]))
print("sumarize of classification")
print(classification_report(y_test,y_predict))


cnf_matrix = confusion_matrix(y_test, y_predict, labels=[1,0])
np.set_printoptions(precision=2) #setting floating point %0.2f
from plot_confusion_matrix import *
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()

#Evaluation with log loss
from sklearn.metrics import log_loss
print(log_loss(y_test, y_predict_prob))


#####################new solver##################
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))