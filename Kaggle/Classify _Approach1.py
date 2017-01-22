import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import sys
import numpy as np
#2.7.12 |Anaconda 4.0.0 (64-bit)|

print (sys.version)
input_file = "E:\\RNotes\\ML\\Pblm2\\TrainDataMultiClassClassification.xls"
df_raw = pd.read_csv(input_file,header=0,sep=",")
print(df_raw.head())
print(df_raw.head(5))

#Remove insignificant id column
df_raw.drop(['Id'],1,inplace=True)
#List all column headers
print(list(df_raw))

print(df_raw.head())

#Fill missing values
df_raw = df_raw.fillna(-999)
df = df_raw

features = list(df.columns[:-1])
print(features);

y = df['class']
x = df[features]

pred_train, pred_test, tar_train, tar_test = train_test_split(x,y,test_size=0.3)
print('Shape of test data')

classifier = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = 40)
classifier = classifier.fit(x,y)

print('acc', classifier.score(x,y))
predictions = classifier.predict(pred_test)
print(predictions)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print('Classifier Accuracy')
print(sklearn.metrics.accuracy_score(tar_test,predictions))

input_file = "E:\\RNotes\\ML\\Pblm2\\TestDataMultiClass.xls"
df_raw = pd.read_csv(input_file,header=0,sep=",")
df2 = pd.read_csv(input_file,header=0,sep=",")
df_raw.drop(['Id'],1,inplace=True)

#from scipy import stats
#df_raw = df_raw[(np.abs(stats.zscore(df_raw)) < 3).all(axis=1)]

#for col in df_raw.columns[:-1]: 
#    meanval = df_raw[col].mean()
#    df_raw[col] = (df_raw[col] - df_raw[col].mean())

#Fill missing values
df_raw = df_raw.fillna(-999)
df = df_raw

x = df[features]
#predictions = rf.predict(x)
predictions = classifier.predict(x)
print('predictions')
#print(predictions)

#i = 0
#for i in range(0,len(predictions)):
#    print(predictions[i])

#print('count',len(predictions))
df['class'] = predictions
#print('count',len(predictions))
df2['class'] = predictions


print('count',df['class'])

#df.to_csv("Results_Multi_Class_Adulteration.csv", sep=',',index=False)
header = ["Id","class"]
df2.to_csv("Results_Multi_Class_Adulteration_Equal_Samples_Sep15.csv", sep=',', columns = header,index=False)
