import pandas as pd
from collections import defaultdict
import sklearn.metrics

fp = open("traindata.txt")
lines = fp.readlines()
#print(lines)

fp = open("trainlabels.txt")
trainlabelslines = fp.readlines()
length = len(trainlabelslines)

fp = open("stoplist.txt")
stopwords = fp.read()
#print(stopwords)

class1 = 0
class0 = 0
for label in trainlabelslines:
    label = label.strip()
    if(label in '1'):
        class1 = class1 + 1
    if(label in '0'):
        class0 = class0 + 1
print(class1)
print(class0)

distincttotalwords = []
for line in lines:
    words=line.split()
    for word in words:
        if word.lower() not in stopwords.lower():
            if word.lower() not in distincttotalwords:
                distincttotalwords.append(word.lower())

distincttotalwords = sorted(distincttotalwords)
print(len(distincttotalwords))
print(distincttotalwords)

list_keys = [ k for k in distincttotalwords]
f = open('Preprocessor.txt', 'w')
keysname = str(list_keys)
keysname = keysname.replace("'", "")
keysname = keysname.strip('[')
keysname = keysname.strip(']')
f.write(keysname+'\n')

for line in lines:
    d = dict.fromkeys(list_keys, 0)
    words=line.split()
    for word in words:
        if(word.lower() in distincttotalwords):
            d[word.lower()] = 1
    list_values = [ v for v in d.values() ]
    keysvalue = str(list_values)
    keysvalue = keysvalue.strip('[')
    keysvalue = keysvalue.strip(']')
    f.write(str(keysvalue)+'\n')
f.close()

d1_docfrequency = {}
d0_docfrequency = {}

j = 0
for line in lines:
    label = trainlabelslines[j] 
    label = label.strip()
    words=line.split()
    for word in words:
        word = word.lower()
        if word in distincttotalwords:
            if word not in d1_docfrequency:
                d1_docfrequency[word] = 1
                d0_docfrequency[word] = 1
            if(label in '1'):            
                 d1_docfrequency[word] += 1
            if(label in '0'):            
                 d0_docfrequency[word] += 1
    j=j+1

d1_prob_frequency = {}
d0_prob_frequency = {}

for word in d1_docfrequency:
     value1 = d1_docfrequency[word]
     value0 = d0_docfrequency[word]
     d1_prob_frequency[word] = value1 / (value0 + value1)
     d0_prob_frequency[word] = value0 / (value0 + value1)

result = []
fp = open("testdata.txt")
testlines = fp.readlines()
#print(testlines)    
i = 0
for line in testlines:
    line = line.strip('\n')
    words=line.split()
    class1prob = class1/(class0+class1)
    class0prob = class0/(class0+class1)
    #print(class1prob)
    #print(class0prob)
    for word in words:
        word = word.lower()
        if word in distincttotalwords:
            class1prob = d1_prob_frequency[word] * class1prob
            class0prob = d0_prob_frequency[word] * class0prob
    if(class1prob > class0prob):
        result.append(1)
    else:
        result.append(0)
    i = i+1
print(result)


input_file = "testlabels.txt"
fp = open(input_file)
testlabelslines = fp.readlines()

i = 0
correct = 0.00
for label in testlabelslines:
    label = label.strip()
    print('label')
    print(label)
    print(result[i])
    if(label in '1'):
        if result[i] == 1:
            correct += 1
    if(label in '0'):
        if result[i] == 0:
            correct += 1
    i = i + 1
    
accuracy = correct / len(result)
print("accuracy")
print(accuracy)