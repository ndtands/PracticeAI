import numpy as np 
import pandas as pd
from matplotlib import colors, pyplot as plt 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets._samples_generator import make_blobs 

X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
'''plt.scatter(X1[:, 0], X1[:, 1], marker='o')
plt.show()'''
#Agglomerative Clustering
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'average')
agglom.fit(X1,y1)

#Visualize data with label
plt.figure(figsize=(6,4))
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)
X1 = (X1 - x_min) / (x_max - x_min)
for i in range(X1.shape[0]):
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),fontdict={'weight': 'bold', 'size': 9})
# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.xticks([])
plt.yticks([])
plt.savefig("img/Hierichical-data-create.png")
plt.show()

#caculate distance matrix
from scipy.spatial import distance_matrix 
dist_matrix = distance_matrix(X1,X1) 


from scipy.cluster import hierarchy 
#Distance between the tow farthest points of two clusters.
# Merge two clusters with the smallest distance
Z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)
plt.savefig("img/Hierichical-with-complete.png")
plt.show()

#Distance between the two closest points in two clusters. 
# Merge two clusters with the smallest distance
Z = hierarchy.linkage(dist_matrix, 'single')
dendro = hierarchy.dendrogram(Z)
plt.savefig("img/Hierichical-with-single.png")
plt.show()

# Merge two clusters with the smallest distance between the centers of threse two clusters
Z = hierarchy.linkage(dist_matrix, 'centroid')
dendro = hierarchy.dendrogram(Z)
plt.savefig("img/Hierichical-with-controid.png")
plt.show()

# Average distance between any two pairs of points in two clusters.
# Merge two clusters with the smallest distance
Z = hierarchy.linkage(dist_matrix, 'average')
dendro = hierarchy.dendrogram(Z)
plt.savefig("img/Hierichical-with-average.png")
plt.show()


######################################################### Clustering on Vehicle dataset #######################################################
filename = 'dataset/cars_clus.csv'
#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna() #Drop NaN
pdf = pdf.reset_index(drop=True) #Reset index
print ("Shape of dataset after cleaning: ", pdf.size)

#Feature selection
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
#normalization
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)




#Clustering using Scipy
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

'''
Essentially, Hierarchical clustering does not require a pre-specified number of clusters. 
However, in some applications we want a partition of disjoint clusters just as in flat clustering. 
So you can use a cutting line:
'''
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
#Also, you can determine the number of clusters directly:
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
fig = pylab.figure(figsize=(12,100))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
plt.savefig("img/hierarchy-scipy.png")
plt.show()


#Clustering using scikit-learn

from sklearn.metrics.pairwise import euclidean_distances
dist_matrix = euclidean_distances(feature_mtx,feature_mtx) 
Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
plt.savefig("img/hierarchy-sklearn.png")
plt.show()

agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(dist_matrix)

agglom.labels_
pdf['cluster_'] = agglom.labels_ #add cluster_


#Visualize
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.savefig("img/Hierichical-dataset-car1.png")
plt.show()

print(pdf.groupby(['cluster_','type'])['cluster_'].count())
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
print(agg_cars)

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.savefig("img/Hierichical-dataset-car.png")
plt.show()