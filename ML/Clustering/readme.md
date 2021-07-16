# 1. K-mean
## 1.1. Algorithm
- Initialize with k centroid randomly
- Caculate distance (SSE,RSSE,..)and assign each point to the closest centroid.
- Copute the new centroids for each cluster by use mean of pre cluster
- repeat unitil centroids no changes
## 1.2. Accuracy
- compare the clusters with the ground trusth if it is avalable
- Caculate average the distance between data points with a centroid of cluster.
## 1.3. Code python
- You can create data with random or load data
```python
# Create data with random
X,y = make_blobs(n_samples=5000,centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# Data actual
cust_df = pd.read_csv("dataset/Cust_Segmentation.csv")
#Pre-processing
df = cust_df.drop('Address', axis=1)
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:] #Loai bo ID
X = np.nan_to_num(X) #Loai bo  NaN
Clus_dataSet = StandardScaler().fit_transform(X)
```
- Main code
```python
from sklearn.cluster import KMeans 
k=4
k_means = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means.fit(X)
k_means_labels= k_means.labels_
k_means_cluster_centers=k_means.cluster_centers_
```
- Visualization 
```python
fig = plt.figure(figsize=(6,4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
# create a plot
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k),colors):
    my_members = (k_means_labels == k) #return True or False of index
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
# Remove x-axis ticks
ax.set_xticks(())  
# Remove y-axis ticks
ax.set_yticks(())
# Show the plot
plt.savefig("img/Kmean-4center.png")
plt.show()''
```
- Elbow method
```python
from scipy.spatial.distance import cdist
'''
- Distortion: It is calculated as the average of the squared distances from the cluster centers of the respective clusters.
Typically, the Euclidean distance metric is used.
'''
distortions =[]
K = range(1, 20)
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.savefig("img/Kmean_elbow.png")
plt.show()
# Caculate the best K
out=[]
for i in range(1,len(distortions)-1):
    a_square = (distortions[i]-distortions[i-1])**2+1
    b_square = (distortions[i]-distortions[i+1])**2+1
    c_square = (distortions[i-1]-distortions[i+1])**2+4
    out.append((a_square+b_square-c_square)/(2*a_square**(1/2)*b_square**(1/2)))

print("K best is : ",out.index(max(out))+2)
```
## 1.4. Disavantage of Kmean algorithm
- you need to how many clusters => You can uses elbow method to find K
- The final solution depends on the initially centers. Some time, you find solution is local minimum.
- clusters need  to have roughly the same number of points
- Can't cluster correctly with the data is round shape or flat shape
## 1.5. Advatage of Kmean algorithm
- It work well with huge amount of data
# 2.Hierarchical clustering 
- Have two type: Agglomerative(bottom-up) and Divisive (top-down)
## 2.1. Agglomerative Algorithm
- Create n clusters, one for each data point
- Compute the proximity matrix [nxn] <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;<&space;Euclid>&space;d(x,y)=(\sum_{i=1}^m|x_{i}-y_{i}|^{r})^{\frac{1}{r}}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;<&space;Euclid>&space;d(x,y)=(\sum_{i=1}^m|x_{i}-y_{i}|^{r})^{\frac{1}{r}}}" title="{\color{White} < Euclid> d(x,y)=(\sum_{i=1}^m|x_{i}-y_{i}|^{r})^{\frac{1}{r}}}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;<Manhattan>&space;d(x,y)=\sum_{i=1}^m|x_{i}-y_{i}|}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;<Manhattan>&space;d(x,y)=\sum_{i=1}^m|x_{i}-y_{i}|}" title="{\color{White} <Manhattan> d(x,y)=\sum_{i=1}^m|x_{i}-y_{i}|}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;<Cosine>&space;cos(\theta)=\frac{\sum_{i=1}^n&space;x_{i}.y_{i}}{\sqrt{\sum_{i=1}^n&space;x_{i}^2}.\sqrt{\sum_{i=1}^n&space;y_{i}^2}}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;<Cosine>&space;cos(\theta)=\frac{\sum_{i=1}^n&space;x_{i}.y_{i}}{\sqrt{\sum_{i=1}^n&space;x_{i}^2}.\sqrt{\sum_{i=1}^n&space;y_{i}^2}}}" title="{\color{White} <Cosine> cos(\theta)=\frac{\sum_{i=1}^n x_{i}.y_{i}}{\sqrt{\sum_{i=1}^n x_{i}^2}.\sqrt{\sum_{i=1}^n y_{i}^2}}}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;<Hamming>&space;}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;<Hamming>&space;}" title="{\color{White} <Hamming> }" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;<Jaccard>&space;d(x,y)=1-\frac{|x\cap&space;y|}{|x\cup&space;y|}&space;}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;<Jaccard>&space;d(x,y)=1-\frac{|x\cap&space;y|}{|x\cup&space;y|}&space;}" title="{\color{White} <Jaccard> d(x,y)=1-\frac{|x\cap y|}{|x\cup y|} }" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{White}&space;<Kullback-Leibler>&space;d(x,y)=\frac{\sum_{i=1}^m&space;x_{i}.log(\frac{x_{i}}{y_{i}})&plus;\sum_{i=1}^m&space;y_{i}.log(\frac{y_{i}}{x_{i}})}{2}&space;;x_{i},y_{i}\neq&space;0}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{\color{White}&space;<Kullback-Leibler>&space;d(x,y)=\frac{\sum_{i=1}^m&space;x_{i}.log(\frac{x_{i}}{y_{i}})&plus;\sum_{i=1}^m&space;y_{i}.log(\frac{y_{i}}{x_{i}})}{2}&space;;x_{i},y_{i}\neq&space;0}" title="{\color{White} <Kullback-Leibler> d(x,y)=\frac{\sum_{i=1}^m x_{i}.log(\frac{x_{i}}{y_{i}})+\sum_{i=1}^m y_{i}.log(\frac{y_{i}}{x_{i}})}{2} ;x_{i},y_{i}\neq 0}" /></a> <br>
- Repeat
    - Merge the two closest cluster with linkage 
        - Linkage complete: Distance between the tow farthest points of two clusters.Merge two clusters with the smallest distance.
        - Linkage single: Distance between the two closest points in two clusters. Merge two clusters with the smallest distance.
        - Linkage centroid: Merge two clusters with the smallest distance between the centers of threse two clusters
        - Linkage average: Average distance between any two pairs of points in two clusters.Merge two clusters with the smallest distance.
    - Update the proximity matrix
- until only a single cluster remains

## 2.2. Code python
### 2.2.1 Data create
- Create data a
```python
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
```
- Agglomerative Clustering
```python
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
# can input parameter is proximity matrix
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
plt.show()
```
- Caculate distance matrix
```python
from scipy.spatial import distance_matrix 
dist_matrix = distance_matrix(X1,X1) 
from scipy.cluster import hierarchy 
# complete, single,average,...
Z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)
plt.show()
```
### 2.2.2. Clustering on Vehicle dataset
- load data and processing
```python
pdf = pd.read_csv(filename)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna() #Drop NaN
pdf = pdf.reset_index(drop=True) #Reset index
#Feature selection
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
#normalization
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
```
- Clustering using scikit-learn
```python
from sklearn.metrics.pairwise import euclidean_distances
dist_matrix = euclidean_distances(feature_mtx,feature_mtx) 
Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
plt.show()
```
- Clustering using scipy
```python
#caculate matrix
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

import pylab
import scipy.cluster.hierarchy
#Merge
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
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'top')
plt.show()
```
- Visualation data with clustering 
```python

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
plt.show()
```

## 2.3. Advantage of Hierarchical Algorithm
- No need to predetermine the number of clusters
- easily implement
- dendrogram
## 2.4. DisAdvantage of Hierarchical Algorithm
- It doesn't work well with huge amount of data
- Log time process

# 3. DBSCAN - Density Based Spatial Clustering of Application with Noise
## 3.1. Algorithm
- you need radius of neighborhood and min number of number
- Find the points in the ε (eps) neighborhood of every point, and identify the core points with more than minPts neighbors.
- Find the connected components of core points on the neighbor graph, ignoring all non-core points.
- Assign each non-core point to a nearby cluster if the cluster is an ε (eps) neighbor, otherwise assign it to noise.
## 3.2. Code python
```python
from sklearn.cluster import DBSCAN 
epsilon = 0.3 #radius
minimumSamples = 7 #Min number of number
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
```
## 3.3. Advantages
- Arbitrarily shaped cluseters
- Robust to outlines
- Does not requre specification of number of clusters

## 3.4 Disavantages
- eps is not easy and cannot be generalized.
- minPts is annot generalize well to clusters of different densities.