
# Density - Based Spatial Clustering of Application with Noise
#import 
import numbers
import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets._samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

# Data genaration
def CreateDatapoints(cenroidLocation, numSamples,CluserDeviation):
    X,y = make_blobs(n_samples=numSamples,centers=cenroidLocation,cluster_std=CluserDeviation)
    #normalize 
    X= StandardScaler().fit_transform(X)
    return X,y

X, y = CreateDatapoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)

#Modeling
'''
Epsilon determine a specified radius that if includes enough number of points within, we call it dense area
minimumSamples determine the minimum number of data points we want in a neighborhood to define a cluster.
'''
epsilon = 0.3 #radius
minimumSamples = 7 #Min number of number
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_

# Distinguish outliers

# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
print(unique_labels)
# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#plot the points with color
for k, col in zip(unique_labels,colors):
    if k==-11:
        col='k'
    class_member_mask=(labels==k) #return index true or flase
    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)
plt.savefig('img/DBSCAN-outline.png')
plt.show()
###############Practice with Kmean 3##################
from sklearn.cluster import KMeans
k=3
k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means3.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
plt.savefig('img/KMean-cant-find-outline.png')
plt.show()


##################### Weather Station Clustering using DBSCAN & scikit-learn ###################
