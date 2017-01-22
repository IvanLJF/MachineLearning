import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

UNCLASSIFIED = False
NOISE = None

def regionset(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        p = m[:,point_id]
        q = m[:,i]
        if((math.sqrt(np.power(p-q,2).sum())) < eps):
            seeds.append(i)
    return seeds

def constructcluster(m, classifications, point_id, cluster_id, eps, min_points):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        p = m[:,point_id]
        q = m[:,i]
        if((math.sqrt(np.power(p-q,2).sum())) < eps):
            seeds.append(i)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = regionset(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def dbscan(m, eps, min_points):
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if constructcluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications

def computedbscan_ds1():
    eps = 0.5
    min_points = 2
    labels_points = dbscan(trans_mat, eps, min_points)
    print(labels_points)
    LABEL_COLOR_MAP = {1 : 'r',
                       2 : 'g',
                      }
    label_color = [LABEL_COLOR_MAP[l] for l in labels_points]
    plt.scatter(trans_mat[0],trans_mat[1], c=label_color)                  
    plt.title("DBSCAN Two Cluster Results")
    plt.show()

def computedbscan_ds2():
    eps = 1
    min_points = 2
    labels_points = dbscan(trans_mat, eps, min_points)
    print(labels_points)
    LABEL_COLOR_MAP = {1 : 'r',
                       2 : 'g',
                       3 : 'b',
                       4 : 'w',
                       5 : 'w',
                       6 : 'c',
                       7 : 'y',
                       None: 'm',
                      }
    label_color = [LABEL_COLOR_MAP[l] for l in labels_points]
    plt.scatter(trans_mat[0],trans_mat[1], c=label_color)                  
    plt.title("DBSCAN Results Dataset 2")
    plt.show()

#read csv
data = pd.read_csv("dataset1.txt", header=None, delimiter=r"\s+")
data_new = data.astype(float)
data_mat = np.matrix(data_new)
trans_mat = np.transpose(data_mat)
computedbscan_ds1()

#read csv
data = pd.read_csv("dataset2.txt", header=None, delimiter=r"\s+")
data_new = data.astype(float)
data_mat = np.matrix(data_new)
trans_mat = np.transpose(data_mat)
computedbscan_ds2()