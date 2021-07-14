#import library needed
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data.head())
print("Shape of data: ",my_data.shape)
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

#preprocesing data
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
#Setting up the Decision Tree

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

#Modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)
#predict
predTree = drugTree.predict(X_testset) #=>ndarray
#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

from sklearn.metrics import classification_report, confusion_matrix
import itertools
print("confusion matrix")
print(confusion_matrix(y_testset,predTree,labels=['drugY', 'drugC', 'drugX', 'drugA', 'drugB']))
print("sumarize of classification")
print(classification_report(y_testset,predTree))


cnf_matrix = confusion_matrix(y_testset, predTree, labels=['drugY', 'drugC', 'drugX', 'drugA', 'drugB'])
np.set_printoptions(precision=2) #setting floating point %0.2f
from plot_confusion_matrix import *
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['drugY', 'drugC', 'drugX', 'drugA', 'drugB'],normalize= False,  title='Confusion matrix')
plt.show()

'''from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
print(featureNames)

targetNames = my_data["Drug"].unique().tolist()
print(targetNames)

out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')'''