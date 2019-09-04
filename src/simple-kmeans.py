'''
Created on Sep 4, 2019

@author: mark
'''
#matplotlib inline
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster
import os

pn=os.path.abspath(__file__)
pn=pn.split("src")[0]
path=os.path.join(pn,'data','home.jpg')

image = ndimage.imread(path)

plt.figure(figsize = (15,8))
plt.imshow(image)

x, y, z = image.shape
image_2d = image.reshape(x*y, z)
image_2d.shape

kmeans_cluster = cluster.KMeans(n_clusters=7)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

plt.figure(figsize = (15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))