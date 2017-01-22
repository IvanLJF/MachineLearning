import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import time
import random
from collections import Counter
import sys
#2.7.12 |Anaconda 4.0.0 (64-bit)

print (sys.version)
start_time = time.time()

#Step 1 - Load Data, Data Cleaning
#Test Data Attached with result set
knndataset = pd.read_csv("Data.txt")

#Replace missing data with outlier value
knndataset.replace('?',-9999,inplace=True)

#Remove insignificant id column
knndataset.drop(['id'],1,inplace=True)

#Identity predictor and dependant values
x = np.array(knndataset.drop(['class'],1))
y = np.array(knndataset['class'])

#Step 2 - Test Data Set Creation
#perform cross_validation and prepare test data
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)

#Step 3 - Buold Classifier
#Invoke decision tree classifier
classifierknn = neighbors.KNeighborsClassifier()
classifierknn.fit(x_train,y_train)
accuracy = classifierknn.score(x_test,y_test)
print('Accuracy Inbuilt function')
print(accuracy)
print("--- %s seconds Inbuilt function ---" % (time.time() - start_time))

#Step 4 - Custom Implementation
#custom function for K nearest neighbours
def k_nearest_neighbors_custom_impl(data,predict,k=3):
    if(len(data)>=k):
        print("K value less than group total")
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes=[i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    #print(vote_result)
    return vote_result

#Step 5 - Custom classifier testing and accuracy computation
start_time = time.time()
df = pd.read_csv("Data.txt")

#Replace missing data with outlier value
df.replace('?',-9999,inplace=True)

#Remove insignificant id column
df.drop(['id'],1,inplace=True)

full_data = df.astype(float).values.tolist()
#print(full_data[:10])
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:+int(test_size*len(full_data))]
test_data =  full_data[:-int(test_size*len(full_data))]
#print("Train data")
#print(train_data)
#print("Test data")
#print(test_data)
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0.0
total = 0.0

for group in test_set:
   # print(group)
    for data in test_set[group]:
        #print(data)
        vote = k_nearest_neighbors_custom_impl(train_set,data,k=5)
        if(group == vote):
            correct+=1
        total+=1
        
print(correct)
print(total)
print('Accuracy custom function', correct/total)
print("--- %s seconds Custom function ---" % (time.time() - start_time))

#Accuracy Inbuilt function
#0.95
#--- 0.0119998455048 seconds Inbuilt function 

#'Accuracy custom function', 0.9553571428571429
#--- 1.39400005341 seconds Custom function ---

