import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import sys
#2.7.12 |Anaconda 4.0.0 (64-bit)|

print (sys.version)
input_file = "Training.xls"
df = pd.read_csv(input_file,header=0)
print(df.head())

print(df.head(5))

#Replace with numerical values
d = {'Male': 1, 'Female':2}
df['Gender'] = df['Gender'].map(d)

#Unique values in column
print(df.Gender.unique())

#Remove insignificant id column
df.drop(['Patient_ID'],1,inplace=True)

print(df.Local_tumor_recurrence.unique())

#List all column headers
print(list(df))

#Fill na as -99
#df.fillna(-99)
#f=df.dropna(axis=1,how='all')

print('unnamed')
print(df['Unnamed: 20'].unique())

#Remove Unnamed Column
for col in df.columns:
    if 'Unnamed' in col:
        del df[col]

#Convert into numerical data
HPV = {'Positive': 1, 'Negative':-1}
df['HPV_p16_status'] = df['HPV_p16_status'].map(HPV)

race = {'White': 1, 'Black':2, 'Hispanic':3, 'Asian':4}
df['Race'] = df['Race'].map(race)

t_side = {'L': 0, 'R':1}
df['Tumor_side'] = df['Tumor_side'].map(t_side)

t_sub_site = {'Tonsil': 0, 'BOT':1, 'Other':2, 'Pharyngeal_wall':3, 'GPS':4, 'Soft_palate':5}
df['Tumor_subsite'] = df['Tumor_subsite'].map(t_sub_site)

print('N Category')
n_category =  {'0':0,'1':1,'2a':2,'3':3,'2c':4,'2b':5}
df['N_category'] = df['N_category'].map(n_category)     

ajcc_stage =  {'II': 0,'III':1,'IV':2,'I':3}
df['AJCC_Stage'] = df['AJCC_Stage'].map(ajcc_stage)     

path_grade = {'III':0, 'II':1, 'NA':2,  'I':3, 'II-III':4, 'IV':5}
df['Pathological_grade'] = df['Pathological_grade'].map(path_grade)     

smok_status = {'Former':1, 'Current':2, 'Never':3,'NA':4}
df['Smoking_status_at_diagnosis'] = df['Smoking_status_at_diagnosis'].map(smok_status)     

print('Induction Chemotherapy')
therapy_unique = {'N':1,'Y':2}
df['Induction_Chemotherapy'] = df['Induction_Chemotherapy'].map(therapy_unique)     

df['Concurrent_chemotherapy'] = df['Concurrent_chemotherapy'].map(therapy_unique)  

print(df.head())

#Fill missing values
df = df.fillna(-99)

#df['Smoking_Pack_Years']= df['Smoking_Pack_Years'].fillna(-1, inplace=True)
#df['Pathological_grade']= df['Pathological_grade'].fillna(-1, inplace=True)
#df['Smoking_Pack_Years'] = df['Smoking_Pack_Years'].map(smoke_years)  

features = list(df.columns[:18])
print(features);

print(df.HPV_p16_status.unique())
print(df.Race.unique())
print(df.Tumor_side.unique())
print(df.Tumor_subsite.unique())
print(df.T_category.unique())
print(df.N_category.unique())
print(df.AJCC_Stage.unique())
print('path')
print(df.Pathological_grade.unique())
print(df.Smoking_status_at_diagnosis.unique())
print('years')
print(df.Smoking_Pack_Years.unique())
print(df.Radiation_treatment_course_duration.unique())
print(df.Total_prescribed_Radiation_treatment_dose.unique())
print(df._Radiation_treatment_fractions.unique())
print(df.Total_prescribed_Radiation_treatment_dose.unique())
print(df.Induction_Chemotherapy.unique())
print(df.KM_Overall_survival_censor.unique())

#fill na with mean values
#df.fillna(df.mean())

y = df['KM_Overall_survival_censor']
x = df[features]

pred_train, pred_test, tar_train, tar_test = train_test_split(x,y,test_size=0.3)
print('Shape of test data')
#print(pred_train)
#print(pred_train.shape)
#print(pred_test)
#print(pred_test.shape)
#print(tar_train)
#print(tar_train.shape)
#print(tar_test)
#print(tar_test.shape)

#Invoke Classifier
classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier = classifier.fit(pred_train,tar_train)

print('acc', classifier.score(x,y))
predictions = classifier.predict(pred_test)
print(predictions)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print('Classifier Accuracy')
print(sklearn.metrics.accuracy_score(tar_test,predictions))

#use max_depth to limit the maximum depth of the tree
#Classifier with 10 Leaf nodes
classifier1 = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = 10)
classifier1 = classifier1.fit(pred_train,tar_train)
predictions = classifier1.predict(pred_test)
print(predictions)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print('Classifier1 Accuracy Leaf Node 10')
print(sklearn.metrics.accuracy_score(tar_test,predictions))

#use max_depth to limit the maximum depth of the tree
#Classifier with 15 Leaf nodes
classifier2 = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = 15)
classifier2 = classifier2.fit(pred_train,tar_train)
predictions = classifier2.predict(pred_test)
print(predictions)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print('Classifier 2 Accuracy Leaf Node 15')
print(sklearn.metrics.accuracy_score(tar_test,predictions))

#use max_depth to limit the maximum depth of the tree
#Classifier with 20 Leaf nodes
classifier3 = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = 20)
classifier3 = classifier3.fit(pred_train,tar_train)
predictions = classifier3.predict(pred_test)
print(predictions)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print('Classifier3 Accuracy Leaf Node 20')
print(sklearn.metrics.accuracy_score(tar_test,predictions))

#Classifier with 25 Leaf nodes
classifier3 = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = 25)
classifier3 = classifier3.fit(pred_train,tar_train)
predictions = classifier3.predict(pred_test)
print(predictions)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print('Classifier3 Accuracy Leaf Node 25')
print(sklearn.metrics.accuracy_score(tar_test,predictions))

#Classifier with 35 Leaf nodes
classifier3 = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = 35)
classifier3 = classifier3.fit(pred_train,tar_train)
predictions = classifier3.predict(pred_test)
print(predictions)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print('Classifier4 Accuracy Leaf Node 35')
print(sklearn.metrics.accuracy_score(tar_test,predictions))

#Cross validation score
crossvalidation = KFold(n=x.shape[0],n_folds=10,shuffle=True,random_state=1)
score = np.mean(cross_val_score(classifier,x,y,scoring='accuracy',cv=crossvalidation,n_jobs=1))
print('score')
print(score)
score = np.mean(cross_val_score(classifier1,x,y,scoring='accuracy',cv=crossvalidation,n_jobs=1))
print('score1')
print(score)
score = np.mean(cross_val_score(classifier2,x,y,scoring='accuracy',cv=crossvalidation,n_jobs=1))
print('score2')
print(score)
score = np.mean(cross_val_score(classifier3,x,y,scoring='accuracy',cv=crossvalidation,n_jobs=1))
print('score3')
print(score)

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
from IPython import display

#Ouput to pdf file
dot_data = StringIO()
tree.export_graphviz(classifier,out_file=dot_data,feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf(r"DecisionTreeResult.pdf")

#max leaves vs Accuracy
import matplotlib.pyplot as plt
x = [10,15,20,25,35]
y = [0.733333333333,0.777777777778,0.8,0.777777777778,0.8,]
plt.xlabel("Number of Leaf Nodes")
plt.ylabel("Accuracy")
plt.plot(x,y,'-o', color = 'g')
plt.title("Decision Tree - Number of Leaf Nodes vs Accuracy")
plt.show()
