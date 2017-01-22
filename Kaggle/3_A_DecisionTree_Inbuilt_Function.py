import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import sys
#2.7.12 |Anaconda 4.0.0 (64-bit)|

print (sys.version)
input_file = "TrainDataBinaryClassification.xls"
df = pd.read_csv(input_file,header=0,sep=",")
print(df.head())
print(df.head(5))

#Remove insignificant id column
df.drop(['Id'],1,inplace=True)
#List all column headers
print(list(df))

print(df.head())

#Fill missing values
df = df.fillna(-99)

features = list(df.columns[:-1])
print(features);

y = df['class']
x = df[features]

pred_train, pred_test, tar_train, tar_test = train_test_split(x,y,test_size=0.3)
print('Shape of test data')

classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier = classifier.fit(pred_train,tar_train)

print('acc', classifier.score(x,y))
predictions = classifier.predict(pred_test)
#print(predictions)
#print(sklearn.metrics.confusion_matrix(tar_test,predictions))
#print('Classifier Accuracy')
#print(sklearn.metrics.accuracy_score(tar_test,predictions))

input_file = "TestDataTwoClassResults.xls"
df = pd.read_csv(input_file,header=0,sep=",")
df2 = pd.read_csv(input_file,header=0,sep=",")
df.drop(['Id'],1,inplace=True)
df = df.fillna(-99)
x = df[features]
predictions = classifier.predict(x)
print('predictions')
#print(predictions)
i = 0
for i in range(0,len(predictions)):
    print(predictions[i])

#print('count',len(predictions))
df2['class'] = predictions

#df.to_csv("Results_Adulteration.csv", sep=',',index=False)
header = ["Id","class"]
df2.to_csv("Results_Adulteration.csv", sep=',', columns = header,index=False)
