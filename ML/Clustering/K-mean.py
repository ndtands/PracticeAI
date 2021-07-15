import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets._samples_generator import make_blobs
np.random.seed(0)

X,y = make_blobs(n_samples=5000,centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
'''print(X.shape)
print(y.shape)'''
#plt.scatter(X[:, 0], X[:, 1], marker='.')
#plt.savefig("img/Kmean-Scatter.png")
k=4
k_means = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means.fit(X)
#print(k_means)
'''k_means_labels= k_means.labels_
k_means_cluster_centers=k_means.cluster_centers_
#print( k_means.labels_)
#print(k_means.cluster_centers_)

# Creating the visual Plot
fig = plt.figure(figsize=(6,4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
# create a plot
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means_labels == k) #return True or False of index
    cluster_center = k_means_cluster_centers[k]
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
# Remove x-axis ticks
ax.set_xticks(())  
# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.savefig("img/Kmean-4center.png")
plt.show()'''

################################## Pratice ##################################
#X,y and K=3
K=3
k_means3 =KMeans(init="k-means++",n_clusters=K,n_init=12)
k_means3.fit(X)
'''fig = plt.figure(figsize=(6,4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k,col in zip(range(K),colors):
    my_members = (k_means3.labels_==k)
    my_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor=col, marker='.')
    ax.plot(my_center[0],my_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.savefig("img/Kmean-3center.png")
plt.show()    '''

#Elbow Method for optimal value of k in Kmeans
from scipy.spatial.distance import cdist
'''
- Distortion: It is calculated as the average of the squared distances from the cluster centers of the respective clusters.
Typically, the Euclidean distance metric is used.
- Inertia: It is the sum of squared distances of samples to their closest cluster center.
=> Choice only one
'''
distortions =[]
inertias =[]
mapping1 = {}
mapping2 = {}
K = range(1, 10)
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    #mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0]
    #mapping2[k] = kmeanModel.inertia_   
'''plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.savefig("img/Kmean_elbow.png")
plt.show()'''

out=[]
for i in range(1,len(distortions)-1):
    a_square = (distortions[i]-distortions[i-1])**2+1
    b_square = (distortions[i]-distortions[i+1])**2+1
    c_square = (distortions[i-1]-distortions[i+1])**2+4
    out.append((a_square+b_square-c_square)/(2*a_square**(1/2)*b_square**(1/2)))

print("K best is : ",out.index(max(out))+2)




################ Customer Segmentation with K-Means #######################
import pandas as pd
cust_df = pd.read_csv("dataset/Cust_Segmentation.csv")
print(cust_df.shape)
#Pre-processing
df = cust_df.drop('Address', axis=1)

#Normalizing over the standard deviation
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:] #Loai bo ID
X = np.nan_to_num(X) #Loai bo  NaN
Clus_dataSet = StandardScaler().fit_transform(X)

#Modeling
clusterNum = 5
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_

#Insights (in details)
df["Clus_km"] = labels #Add column
print(df.groupby('Clus_km').mean())

#Visualize
area = np.pi * ( X[:, 1])**2  
#X[:,0] :Age , X[:3]:icome, X[:1]: Education, Color is labels
'''plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float64), alpha=0.5,) 
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.savefig("img/Income and education.png")
plt.show()'''

# Visualize
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float64))
#plt.savefig("img/Kmean-income.png")
plt.show()
'''
- AFFLUENT, EDUCATED AND OLD AGED
- MIDDLE AGED AND MIDDLE INCOME
- YOUNG AND LOW INCOME
'''

# Evaluation
distortions =[]
inertias =[]
mapping1 = {}
mapping2 = {}
K = range(1, 10)
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.savefig("img/Kmean_elbow-income.png")
plt.show()
out=[]
for i in range(1,len(distortions)-1):
    a_square = (distortions[i]-distortions[i-1])**2+1
    b_square = (distortions[i]-distortions[i+1])**2+1
    c_square = (distortions[i-1]-distortions[i+1])**2+4
    out.append((a_square+b_square-c_square)/(2*a_square**(1/2)*b_square**(1/2)))

print("K best is : ",out.index(max(out))+2)
