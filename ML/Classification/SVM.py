import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head())
#ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
#cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
#plt.show()

#preprocessing data
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

#Split data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
# Modeling


###############################Linear#####################################
from sklearn import svm
#clf = svm.SVC(kernel='rbf')
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
print("w =", clf.coef_,"\n","b =", clf.intercept_)
yhat = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

#print (classification_report(y_test, yhat))
from plot_confusion_matrix import *
# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
#plt.savefig("linear_SVM.png")
#plt.show()

print("Evaluate Linear")
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat,pos_label=2))

###############################Polynomial#####################################
from sklearn import svm
#clf = svm.SVC(kernel='rbf')
clf = svm.SVC(kernel = 'poly',degree=3)
clf.fit(X_train, y_train)
print("b =", clf.intercept_)
yhat = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

#print (classification_report(y_test, yhat))
from plot_confusion_matrix import *
# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
#plt.savefig("poly_SVM.png")
#plt.show()

print("Evaluate Poly")
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat,pos_label=2))

###############################Radial basis function#####################################
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
print("b =", clf.intercept_)
yhat = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

#print (classification_report(y_test, yhat))
from plot_confusion_matrix import *
# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
#plt.savefig("Radial basis function_SVM.png")
#plt.show()

print("Evaluate Radial basis function")
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat,pos_label=2))


################################Sigmoid####################################

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn import svm
clf = svm.SVC(kernel='sigmoid')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

clf.fit(X_train, y_train)
print("b =", clf.intercept_)
yhat = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)
#print (classification_report(y_test, yhat))
from plot_confusion_matrix import *
# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
#plt.savefig("sigmoid_SVM.png")
#plt.show()

print("Evaluate sigmoid")
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat,pos_label=2))
