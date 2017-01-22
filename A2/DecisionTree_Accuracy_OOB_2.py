from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from mlxtend.classifier import EnsembleVoteClassifier
import sklearn.metrics
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

input_file = "spam.data.txt"
df_raw = pd.read_csv(input_file,header=None,sep=' ')
print(df_raw.head())
df = df_raw
features = list(df.columns[:-1])

y = df[57]
x = df[features]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)

accuracy = []
featurecount = []
for i in range(10,55):
    clf1 = RandomForestClassifier(warm_start=True,max_features=i)
    clf1.fit(x_train,y_train)
    predictions4 = clf1.predict(x_test)
    accuracy.append(sklearn.metrics.accuracy_score(y_test,predictions4))
    featurecount.append(i)

for i in range(0,len(featurecount)):
    print('Feature count is ',featurecount[i], ' - Accuracy is -', accuracy[i])
    #print(max(accuracy))
    if(accuracy[i]==max(accuracy)):
        val = i

print('Max Val of Accuracy with Features')        
print(featurecount[val])
print('Accuracy Obtained')
print(accuracy[val])

oob_score = []
ivalue = []
accuracyobtained = []
for i in range(10,55):
    clf1 = RandomForestClassifier(warm_start=True,max_features=val,oob_score=True,n_estimators= i)
    clf1.fit(x_train,y_train)
    predictions = clf1.predict(x_test)
    Error = 1-(sklearn.metrics.accuracy_score(y_test,predictions))
    print('Error')
    print(Error)
    accuracyobtained.append(Error)   
    ivalue.append(i)
    oob_score.append(1 - clf1.oob_score_)

print('accuracyobtained')
print(accuracyobtained)

j = 0
for i in range(10,55):
    plt.xlim(10, 55)
    plt.ylim(0.03, 0.1)
    plt.plot(i,oob_score[j],'ro',color='g')
    plt.plot(i,accuracyobtained[j],'ro',color='r')    
    plt.legend(loc="upper right")
    plt.xlabel("n_estimators ")
    plt.ylabel("OOB error - green, test error - red")
    plt.text(40, 0.09, 'OOB - GREEN')
    plt.text(40, 0.08, 'TEST ERROR - RED')    
    j = j+1
 
plt.show()
print(accuracyobtained)