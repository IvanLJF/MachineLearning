import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
import sys
import numpy as np
from sklearn import svm

print (sys.version)
input_file = "E:\\RNotes\\ML\\Pblm2\\TrainDataMultiClassClassification_Custom_sep18.csv"
df_raw = pd.read_csv(input_file,header=0,sep=",")

#Remove insignificant id column
df_raw.drop(['Id'],1,inplace=True)

df_raw = df_raw.fillna(-999)
df = df_raw

features = list(df.columns[:-1])
print(features);

y = df['class']
x = df[features]
 
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x,y)
 
input_file = "E:\\RNotes\\ML\\Pblm2\\TestDataMultiClass.xls"
df = pd.read_csv(input_file,header=0,sep=",")
df2 = pd.read_csv(input_file,header=0,sep=",")
df.drop(['Id'],1,inplace=True)
df = df.fillna(-999)
x = df[features]

predictions = clf.predict(x)

i = 0
for i in range(0,len(predictions)):
    print(predictions[i])

df['class'] = predictions
df2['class'] = predictions
print('count',df['class'])

header = ["Id","class"]
df2.to_csv("Results_Multi_Class_Adulteration_sep18_SVC1.csv", sep=',', columns = header,index=False)
