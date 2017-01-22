import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR

data = pd.read_csv('train.csv',header=0)
#print(data.head())
y = data.loss.values
features = list(data.columns[:-1])
print(y)


for col in data.columns[:-1]: 
    print(data[col].unique())
    
#Each column Find Cat Variables
#Rename as Cat Variable + ith position of column

i = 1
for col in data.columns[:-1]: 
    if('cat' in str(col)):
        data[col] = data[col] + str(i)
    i = i+1

i = 0
for col in data.columns[:-1]: 
    if('cat' in str(col)):
        df_newframe = pd.get_dummies(data[col])
        col = 'new'+str(i)+str(col)
        i = i+1
        data = pd.concat([data, df_newframe], axis=1)
        
i = 1
for col in data.columns[:-1]: 
    if('cat' in str(col)):
            if('new' not in str(col)):
                data = data.drop(col,1)

y = np.log(data.loss.values)
data_new = data.drop('id',1)
data_new = data.drop('loss',1)
data_new =  data_new.astype(float) 
features = list(data_new.columns[:-1])
datatest = pd.read_csv('test.csv',header=0)

i = 1
for col in datatest.columns: 
    if('cat' in str(col)):
        datatest[col] = datatest[col] + str(i)
    i = i+1

i = 1
for col in datatest.columns: 
    if('cat' in str(col)):
        df_newtestframe = pd.get_dummies(datatest[col])
        col = 'new'+str(i)+str(col)
        i = i+1
        datatest = pd.concat([datatest, df_newtestframe], axis=1)

i = 1
for col in datatest.columns: 
    if('cat' in str(col)):
        if('new' not in str(col)):
            datatest = datatest.drop(col,1)
        
#Add missing columns
for coldata in data_new.columns:
    if coldata not in datatest.columns:
        datatest[coldata] = 0

for coldata in datatest.columns:
    if coldata not in data_new.columns:
        data_new[coldata] = 0

print('Header Rows')
data_new = data_new.drop('id',1)
print(data_new.head())

datatest = datatest.drop('id',1)
features = list(data_new.columns)

print('Header Test Rows')
print(datatest.head())

#dictvalues ={}
#for coldata in data_new.columns:
#    dictvalues[coldata] = datatest[coldata].mean()

#print('dictvalues values')
#print(dictvalues)

##print('sorted output')
#from operator import itemgetter
#print(sorted(dictvalues.items(), key=itemgetter(1),reverse=True))

#regr = linear_model.Lasso(alpha=0.1)
regr = LinearSVR(C=1.0, epsilon=0.2)
#regr = RandomForestRegressor()
#regr = AdaBoostRegressor(n_estimators=80)
regr.fit(data_new[features], y)

predictions = regr.predict(datatest)
print('predictions')
print(predictions)

datatest_result = pd.read_csv('test.csv',header=0)
datatest_result['loss'] = np.exp(predictions)
header = ["id","loss"]
datatest_result.to_csv("Results_AllState_SVR_81.csv", sep=',', columns = header,index=False)

for col in data.columns[:-1]: 
    print(data[col].unique())

