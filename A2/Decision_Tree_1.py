import numpy as np
import random
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import time

input_file = "spam.data.txt"
df_raw = pd.read_csv(input_file,header=None,sep=' ')
print(df_raw.head())
df = df_raw
features = list(df.columns[:-1])
print(features);
print(len(features))

start = time.clock()
y = df[57]
x = df[features]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

rows = np.random.choice(df.index.values, 600)
d1 = df.ix[rows]

a = []
for i in range(1,56):
    a.append(i)

feature2 = random.sample(a,15)
y1 = []
y1 = d1[57]
#print(y1)
print(feature2)
for col in d1.columns:
    #print(col)
    if(col not in feature2):
        df1 = d1.drop(col,1)
#print(df1)
#print(y1)
clf1 = tree.DecisionTreeClassifier()
clf1.fit(df1,y1)
            
rows = np.random.choice(df.index.values, 600)
d2 = df.ix[rows]
feature2 = random.sample(a,15)
y2 = []
y2 = d2[57]
print(feature2)
for col in d1.columns:
    #print(col)
    if(col not in feature2):
        df2 = d2.drop(col,1)
#print(df2)
#print(y2)
clf2 = tree.DecisionTreeClassifier()
clf2.fit(df2,y2)

rows = np.random.choice(df.index.values, 600)
d3 = df.ix[rows]
feature2 = random.sample(a,15)
y3 = []
y3 = d3[57]
print(feature2)
for col in d3.columns:
    #print(col)
    if(col not in feature2):
        df3 = d3.drop(col,1)
#print(df3)
#print(y3)
clf3 = tree.DecisionTreeClassifier()
clf3.fit(df3,y3)

rows = np.random.choice(df.index.values, 600)
d4 = df.ix[rows]
feature2 = random.sample(a,15)
y4 = []
y4 = d4[57]
print(feature2)
for col in d4.columns:
    #print(col)
    if(col not in feature2):
        df4 = d4.drop(col,1)
#print(df4)
#print(y4)
clf4 = tree.DecisionTreeClassifier()
clf4.fit(df4,y4)

rows = np.random.choice(df.index.values, 600)
d5 = df.ix[rows]
feature2 = random.sample(a,15)
y5 = []
y5 = d5[57]
print(feature2)
for col in d5.columns:
    #print(col)
    if(col not in feature2):
        df5 = d5.drop(col,1)
#print(df5)
#print(y5)
clf5 = tree.DecisionTreeClassifier()
clf5.fit(df5,y5)

rows = np.random.choice(df.index.values, 600)
d6 = df.ix[rows]
feature2 = random.sample(a,15)
y6 = []
y6 = d6[57]
print(feature2)
for col in d6.columns:
    #print(col)
    if(col not in feature2):
        df6 = d6.drop(col,1)
#print(df6)
#print(y6)
clf6 = tree.DecisionTreeClassifier()
clf6.fit(df6,y6)

rows = np.random.choice(df.index.values, 600)
d7 = df.ix[rows]
feature2 = random.sample(a,15)
y7 = []
y7 = d7[57]
#print(feature2)
for col in d7.columns:
    #print(col)
    if(col not in feature2):
        df7 = d7.drop(col,1)
#print(df7)
#print(y7)
clf7 = tree.DecisionTreeClassifier()
clf7.fit(df7,y7)

rows = np.random.choice(df.index.values, 600)
d8 = df.ix[rows]
feature2 = random.sample(a,15)
y8 = []
y8 = d8[57]
print(feature2)
for col in d8.columns:
    #print(col)
    if(col not in feature2):
        df8 = d8.drop(col,1)
#print(df8)
#print(y8)
clf8 = tree.DecisionTreeClassifier()
clf8.fit(df8,y8)

rows = np.random.choice(df.index.values, 600)
d9 = df.ix[rows]
feature2 = random.sample(a,15)
y9 = []
y9 = d9[57]
print(feature2)
for col in d9.columns:
    #print(col)
    if(col not in feature2):
        df9 = d9.drop(col,1)
#print(df9)
#print(y9)
clf9 = tree.DecisionTreeClassifier()
clf9.fit(df9,y9)

rows = np.random.choice(df.index.values, 600)
d10 = df.ix[rows]
feature2 = random.sample(a,15)
y10 = []
y10 = d10[57]
print(feature2)
for col in d10.columns:
    #print(col)
    if(col not in feature2):
        df10 = d10.drop(col,1)
#print(df10)
#print(y10)
clf10 = tree.DecisionTreeClassifier()
clf10.fit(df10,y10)

predictions1 = clf1.predict(x_test)
print(len(predictions1))
predictions2 = clf2.predict(x_test)
print(len(predictions2))
predictions3 = clf3.predict(x_test)
print(len(predictions3))
predictions4 = clf4.predict(x_test)
print(len(predictions4))
predictions5 = clf5.predict(x_test)
print(len(predictions5))
predictions6 = clf6.predict(x_test)
print(len(predictions6))
predictions7 = clf7.predict(x_test)
print(len(predictions7))
predictions8 = clf8.predict(x_test)
print(len(predictions8))
predictions9 = clf9.predict(x_test)
print(len(predictions9))
predictions10 = clf10.predict(x_test)
print(len(predictions10))

#merge results
df = pd.DataFrame({1:predictions1,2:predictions1,3:predictions3,4:predictions4,5:predictions5,6:predictions6,7:predictions7,8:predictions8,9:predictions9,10:predictions10})
print(df)

#find maximum occuring values
results = pd.DataFrame(df.mode(axis=1))
print(results[0])
print('Time for custom code -',time.clock() - start)
print('Classifier Accuracy - Decision Tree based Custom Code -',sklearn.metrics.accuracy_score(y_test,results[0]))

start = time.clock()
clf1 = RandomForestClassifier()
clf1.fit(x_train,y_train)
predictions1 = clf1.predict(x_test)
print('Classifier Accuracy - Random Forest Function - ',sklearn.metrics.accuracy_score(y_test,predictions1))
print('Time for Random Forest code - ',time.clock() - start)
