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
c = a*b

weights=(b*a+(0.5)*np.identity(a.shape[1])).I*b*y_mat
print('weights')
   
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

print('length')
print(length)
print('diameter')
print(diameter)
print('height')
print(height)
print('whole_weight')
print(whole_weight)
print('shucked_weight')
print(shucked_weight)
print('viscera_weight')
print(viscera_weight)
print('shell_weight')
print(shell_weight)

#Split data for training and testing set
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.6)
print(len(x_test))
print(len(y_test))
x_test_new = x_test.values.astype(np.float)

sumofsquares = 0
i = 0
for row in x_test_new:
    result = (row[0]*length) + (row[1]*diameter) + (row[2]*height) + (row[3]*whole_weight) + (row[4]*shucked_weight) + (row[5]*viscera_weight) + (row[6]*shell_weight) + (row[7]*M) + (row[8]*F) + (row[9]*I) 
    sumofsquares = sumofsquares + (y_test[i]-result)*(y_test[i]-result)
    i=i+1

print(len(y_test))
print('sumofsquares')
print(sumofsquares)
print('avg sum of squares')
print(sumofsquares/len(y_test))
mse = []
lamdavals = []

x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.1)
print(len(x_test))
print(len(y_test))
x_test_new = x_test.values.astype(np.float)
 
for i in range(1,15):
    lambdaval = i/2
    lamdavals.append(lambdaval)
    weights=(b*a+(lambdaval)*np.identity(a.shape[1])).I*b*y_mat
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
        sumofsquares = sumofsquares + (y_test[i]-result)*(y_test[i]-result)
        i=i+1
    mse.append(sumofsquares/len(y_test))

print('minimum of mse')
print(min(mse))

minvalerror = min(mse)

print('mse length')
print(len(mse))
for i in range(0,len(mse)):
    print('lambda value',lamdavals[i])
    print('mean squared error',mse[i])
    if(minvalerror == mse[i]):
        lambvalopt = lamdavals[i]

print('best lambda value -',lambvalopt)
print('mse  error value -',minvalerror)
#best lambda value
#0.5
#minimum error value
#[[ 4.75310998]]
predictedvalues = []
def mylinridgereg(a,b,y_mat,lambdavalueresult):
    #Remove two weights
    weights=(b*a+(lambdavalueresult)*np.identity(a.shape[1])).I*b*y_mat
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
        predictedvalues.append(result)      
        sumofsquares = sumofsquares + (y_test[i]-result)*(y_test[i]-result)
        i=i+1
    
    print('Linear Regression')
    print('sumofsquares - ',sumofsquares)
    print('mse -',sumofsquares/len(y_test))

print('Linear Regression Function Call')
mylinridgereg(a,b,y_mat,lambvalopt)

#Plot for predicted vs actual value
print('result plot starts here')
print('best lambda value is ',lambvalopt)
print(len(predictedvalues))
print(len(y_test))
plt.figure()
plt.xlim(0,30)
plt.ylim(0,30)
for i in range(0,len(y_test)):
    plt.plot(predictedvalues[i],y_test[i],'ro',color='g')
    plt.xlabel("predicted values")
    plt.ylabel("Actual values")    
plt.title(" predicted values vs Actual values")
plt.show()

def mylinridgeregRemovedFeatures(a,b,y_mat,lambdavalueresult):
    #Remove two weights
    weights=(b*a+(lambdavalueresult)*np.identity(a.shape[1])).I*b*y_mat
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
    height = 0
    diameter = 0
    
    sumofsquares = 0
    i = 0
    for row in x_test_new:
        result = (row[0]*length) + (row[1]*diameter) + (row[2]*height) + (row[3]*whole_weight) + (row[4]*shucked_weight) + (row[5]*viscera_weight) + (row[6]*shell_weight) + (row[7]*M) + (row[8]*F) + (row[9]*I) 
        sumofsquares = sumofsquares + (y_test[i]-result)*(y_test[i]-result)
        i=i+1
    
    print('After removing two weights')
    print('sumofsquares -', sumofsquares)
    print('mse - ',sumofsquares/len(y_test))
    
print('Linear Regression Function Call - Removed Params')
mylinridgeregRemovedFeatures(a,b,y_mat,0.5)
