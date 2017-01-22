
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('train.csv',header=0)
data = data.drop('Y',1)
data = data.drop('Time',1)
colnames = list(data.columns)
print(colnames)
data1 = data.transpose()
print(data1.head())

for col in data1.columns: 
    if('cat' in str(col)):
        data1[col] = data1[col]-data1[col].mean()
    
kmeans = KMeans(n_clusters=8, random_state=0).fit(data1)

node1 = []
node2 = []
node1 = data1[0]
node2 = kmeans.labels_

dataresult = pd.DataFrame()
dataresult['Node'] = colnames
dataresult['Cluster'] = node2

dataresult = dataresult[dataresult.Node != 'Y']
dataresult = dataresult[dataresult.Node != 'Time']
print(dataresult)
dataresult.to_csv("Results_1.csv", sep=',',index=False)