import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('linregdata.txt',header=None,names=['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings'])

for label in "MFI":
        data[label] = data["sex"] == label
del data["sex"]

#Standardize columns
data["diameter"] = (data["diameter"]-data["diameter"].mean())/np.std(data["diameter"])
#height = 
data["height"] = (data["height"]-data["height"].mean())/np.std(data["height"])
#whole_weight =  
data["whole_weight"] = (data["whole_weight"]-data["whole_weight"].mean())/np.std(data["whole_weight"])
#shucked_weight =  
data["shucked_weight"] = (data["shucked_weight"]-data["shucked_weight"].mean())/np.std(data["shucked_weight"])
#viscera_weight =  
data["viscera_weight"] = (data["viscera_weight"]-data["viscera_weight"].mean())/np.std(data["viscera_weight"])
#shell_weight = 
data["shell_weight"] = (data["shell_weight"]-data["shell_weight"].mean())/np.std(data["shell_weight"])

y = data.rings.values
y_new = y.astype(float)
y_mat = np.matrix(y_new)
y_mat = np.transpose(y_mat)
del data["rings"]

data_x = data.copy(deep=True)
data_y = y

print(data.head())
x = data.values.astype(np.float)
x_new = x.astype(float)
x_transpose = np.transpose(x_new)
a = np.matrix(x_new)
b = np.matrix(x_transpose)

mse = []
lamdavals = []

partitionsize = []
trainingresultmse = []
testresultmse = []
lambdatraininga = []
lambdatestinga = []

for part in range(1,5):
    part = part/10
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=part)
    partitionsize.append(part)
    
    for m in range(1,26):
        x_traina, x_testa, y_traina, y_testa = train_test_split(x_train,y_train,test_size=0.8)
        x1 = x_traina.values.astype(np.float)  
        x_new1 = x1.astype(float)    
        a1 = np.matrix(x_new1)
        b1 = np.matrix(np.transpose(a1))
        y_new1 = y_traina.astype(float)
        y_mat1 = np.matrix(y_new1)
        y_mat1 = np.transpose(y_mat1)

        m = m+1
        #print(len(x_testa))
        #print(len(y_testa))
        x_test_new = x_testa.values.astype(np.float)
        #print('part')
        #print(part)
        for i in range(1,15):
            lambdaval = i/2
            lamdavals.append(lambdaval)
            weights=(b1*a1+(lambdaval)*np.identity(a1.shape[1])).I*b1*y_mat1
            length = weights[0]
            diameter = weights[1]
            height = weights[2]
            whole_weight = weights[3]
            shucked_weight = weights[4]
            viscera_weight = weights[5]
            shell_weight = weights[6]
            M = weights[7]
            F = weights[8]
            I = weights[9]
        
            #Split data for training and testing set
            sumofsquares = 0
            i = 0
            for row in x_test_new:
                result = (row[0]*length) + (row[1]*diameter) + (row[2]*height) + (row[3]*whole_weight) + (row[4]*shucked_weight) + (row[5]*viscera_weight) + (row[6]*shell_weight) + (row[7]*M) + (row[8]*F) + (row[9]*I) 
                sumofsquares = sumofsquares + (y_testa[i]-result)*(y_testa[i]-result)
                i=i+1
            mse.append(sumofsquares/len(y_testa))
        
        minvalerror = min(mse)
        
        for i in range(0,len(mse)):
            if(minvalerror == mse[i]):
                lambvalopt = lamdavals[i]
        lambdatraininga.append(lambvalopt)        
       
        weights=(b1*a1+(lambvalopt)*np.identity(a1.shape[1])).I*b1*y_mat1
        #print('weights')
           
        length = weights[0]
        diameter = weights[1]
        height = weights[2]
        whole_weight = weights[3]
        shucked_weight = weights[4]
        viscera_weight = weights[5]
        shell_weight = weights[6]
        M = weights[7]
        F = weights[8]
        I = weights[9]
        sumofsquares = 0
        i = 0
        for row in x_test_new:
            result = (row[0]*length) + (row[1]*diameter) + (row[2]*height) + (row[3]*whole_weight) + (row[4]*shucked_weight) + (row[5]*viscera_weight) + (row[6]*shell_weight) + (row[7]*M) + (row[8]*F) + (row[9]*I) 
            sumofsquares = sumofsquares + (y_testa[i]-result)*(y_testa[i]-result)
            i=i+1
        
        trainingresultmse.append(sumofsquares/len(y_testa))

    for m in range(1,26):

        x_trainb, x_testb, y_trainb, y_testb = train_test_split(x_test,y_test,test_size=part)        
        x2 = x_trainb.values.astype(np.float)  
        x_new2 = x2.astype(float)    
        a1 = np.matrix(x_new2)
        b1 = np.matrix(np.transpose(a1))
        y_new2 = y_trainb.astype(float)
        y_mat1 = np.matrix(y_new2)
        y_mat1 = np.transpose(y_mat1)

        m = m+1
        #print(len(x_test))
        #print(len(y_test))
        x_test_new = x_testb.values.astype(np.float)
        #print('part')
        #print(part)
        for i in range(1,15):
            lambdaval = i/2
            lamdavals.append(lambdaval)
            weights=(b1*a1+(lambdaval)*np.identity(a1.shape[1])).I*b1*y_mat1
            length = weights[0]
            diameter = weights[1]
            height = weights[2]
            whole_weight = weights[3]
            shucked_weight = weights[4]
            viscera_weight = weights[5]
            shell_weight = weights[6]
            M = weights[7]
            F = weights[8]
            I = weights[9]
        
            #Split data for training and testing set
            sumofsquares = 0
            i = 0
            for row in x_test_new:
                result = (row[0]*length) + (row[1]*diameter) + (row[2]*height) + (row[3]*whole_weight) + (row[4]*shucked_weight) + (row[5]*viscera_weight) + (row[6]*shell_weight) + (row[7]*M) + (row[8]*F) + (row[9]*I) 
                sumofsquares = sumofsquares + (y_testb[i]-result)*(y_testb[i]-result)
                i=i+1
            mse.append(sumofsquares/len(y_testb))
        
        minvalerror = min(mse)
        for i in range(0,len(mse)):
            if(minvalerror == mse[i]):
                lambvalopt = lamdavals[i]
                print('optimum lambda value-', lambvalopt)
        
        lambdatestinga.append(lambvalopt)
        
        weights=(b1*a1+(lambvalopt)*np.identity(a1.shape[1])).I*b1*y_mat1
           
        length = weights[0]
        diameter = weights[1]
        height = weights[2]
        whole_weight = weights[3]
        shucked_weight = weights[4]
        viscera_weight = weights[5]
        shell_weight = weights[6]
        M = weights[7]
        F = weights[8]
        I = weights[9]
        sumofsquares = 0
        i = 0
        for row in x_test_new:
            result = (row[0]*length) + (row[1]*diameter) + (row[2]*height) + (row[3]*whole_weight) + (row[4]*shucked_weight) + (row[5]*viscera_weight) + (row[6]*shell_weight) + (row[7]*M) + (row[8]*F) + (row[9]*I) 
            sumofsquares = sumofsquares + (y_testb[i]-result)*(y_testb[i]-result)
            i=i+1
        
        testresultmse.append(sumofsquares/len(y_testb))

print('No of Partitions')
print(len(partitionsize))  

print('Train MSE')
print('length of training')       
print(len(trainingresultmse)) 
    
print('Test MSE')
print('length of testing data')       
print(len(testresultmse)) 

print('lamda value length')
print(len(lambdatraininga))

plt.figure()
plt.xlim(0.0, 0.7)
plt.ylim(0.0,10.0)
x = [0.1,0.2,0.3,0.4]
k = 0
for i in range(0,4):
    for j in range(1,25):
        plt.plot(x[i],testresultmse[k],'ro',color='g')
        k = k+1
        plt.xlabel("parition values")
        plt.ylabel("min mse testing ")    
plt.title(" Partition values vs min mse")
plt.show()

print('lambda')
print(lambdatestinga)
print(len(lambdatestinga))

plt.figure()
plt.xlim(0.0, 8)
plt.ylim(0.0,10.0)
for i in range(0,100):
    plt.plot(lambdatestinga[i],testresultmse[k],'ro',color='g')
    plt.xlabel("optimum lambda values")
    plt.ylabel("min mse testing ")    
plt.title("lambda values vs min mse")
plt.show()



