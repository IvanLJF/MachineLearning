import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
distance = [0.12, 0.51, 0.84, 0.28, 0.34, 0.25, 0.16, 0.77, 0.61, 0.14, 0.7, 0.68, 0.45, 0.2, 0.67]
linkage_matrix = linkage(distance, "single")
plt.title("Hierarchical clustering  - Single Linkage")
dendrogram(linkage_matrix,
           color_threshold=1,
           truncate_mode='lastp',
           labels=['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
           ,
           distance_sort='descending'
           )
plt.show()




linkage_matrix = linkage(distance, "complete")
plt.title("Hierarchical clustering  - Complete Linkage")
dendrogram(linkage_matrix,
           color_threshold=1,
           truncate_mode='lastp',
           labels=['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
           ,
           distance_sort='ascending'
           )
plt.show()