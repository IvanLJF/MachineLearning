
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, neighbors, linear_model

data = pd.read_csv('train.csv',header=0)
#lb = LabelEncoder()
#y = lb.fit_transform(data.Y.values)

for coldata in data.columns[:-1]:
        data[coldata] = data[coldata] - data[coldata].mean()

print(data.Y.unique())
#print(data.Y.values)
#print(y)
#data = data.fillna(-1)
#print(data.head())
features = list(data.columns[:-1])
print(features)
#resultdata = map(int, data.Y.values)
#print(resultdata)

#clf = svm.SVC()
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(data[features],data.Y.values) 

datatest = pd.read_csv('test.csv',header=0)

for coldata in datatest.columns:
        datatest[coldata] = datatest[coldata] - datatest[coldata].mean()

predictions = clf.predict(datatest)
print(predictions)

datatest_result = pd.read_csv('test.csv',header=0)
datatest_result['Y'] = predictions
header = ["Time","Y"]
datatest_result.to_csv("HE_PREDICT1.csv", sep=',', columns = header,index=False)
