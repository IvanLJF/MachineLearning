from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#create sample test circles
data = make_circles()

X = data[0]
plt.figure(1)
plt.scatter(X[:,0],X[:,1])

#Apply Kmeans cluster with K value 2
model = KMeans(n_clusters=2)
model.fit(X)

labels = model.labels_
print labels
plt.figure(2)
#Plot the cluster
plt.scatter(X[:,0],X[:,1],c=labels)

#Plot Center Values of it
print model.cluster_centers_
plt.show()