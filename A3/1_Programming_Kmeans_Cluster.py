
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#read csv
data = pd.read_csv("dataset1.txt", header=None, delimiter=r"\s+")
print(data.head())
#Do K means with cluster = 2 and cluster 3
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

#Plot Results Cluster = 2
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'g',
                  }
label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
plt.scatter(data[0],data[1], c=label_color)                  
plt.title("Two Cluster Results - DataSet 1 - Kmeans")
plt.show()

#read csv
data = pd.read_csv("dataset2.txt", header=None, delimiter=r"\s+")
print(data.head())
kmeans = KMeans(n_clusters=7, random_state=0).fit(data)

#Plot Results Cluster = 7
LABEL_COLOR_MAP = {    0 : 'r',
                       1 : 'g',
                       2 : 'b',
                       3 : 'w',
                       4 : 'w',
                       5 : 'c',
                       6 : 'y',
                       None: 'm',
                      }
label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
plt.scatter(data[0],data[1], c=label_color)                  
plt.title("Seven Cluster Results Data Set 2 - Kmeans")
plt.show()

