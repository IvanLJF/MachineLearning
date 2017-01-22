import matplotlib.pyplot as plt
import os
import fnmatch
import numpy as np
from sklearn.decomposition import PCA
import scipy.misc
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy.linalg as linalg
import sys
from PIL import Image

matches = []
for root, dirs, files in os.walk("./data/faces"):
    for filename in fnmatch.filter(files, '*.png'):
        matches.append(os.path.join(root, filename))
        data = []
        for m in matches:
            data.append(plt.imread(m).flatten())

data_matrix = np.matrix(np.column_stack(data))
print(data_matrix.shape)

#Mean of Each row
#Difference Phase - Subtrach mean from Each Column
# Recent versions of numpy
diff_matrixmean = data_matrix - data_matrix.mean(axis=1)
print(diff_matrixmean.shape)

#Difference Phase Xply by Difference Phase Transpose
diff_matrixmeantranspose = np.transpose(diff_matrixmean)
print(diff_matrixmeantranspose.shape)

covariancematrix = diff_matrixmeantranspose*diff_matrixmean
print(covariancematrix.shape)

#Eigen Decomposition
eigenValues,eigenVectors = linalg.eig(covariancematrix)

#Sort them
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]

print(eigenVectors.shape)

#Part B
#Normalize and Plot
Nvalues = eigenValues/eigenValues[0]
print('Normalized Values')
print(Nvalues)

#From Eigen Values compute 90/80/50
values = eigenValues
valsum = 0
valpercentage = []
for value in values:
    valsum = value+valsum
    #print(value)
    
for value in values:
    currentpercentage = 0
    currentpercentage = (value/valsum)*100
    valpercentage.append(currentpercentage)

#print(valpercentage)
    
c90 = 0
c80 = 0
c50 = 0

i = 0
currentval = 0
for value in valpercentage:
    i = i+1
    currentval = currentval+ value
    if(currentval > 50 and c50 == 0):
        c50 = i
    if(currentval > 80 and c80 == 0):
        c80 = i    
    if(currentval > 90 and c90 == 0):
        c90 = i    

print('c90',c90)
print('c80',c80)
print('c50',c50)

plt.xlim(0, 50)
plt.ylim(0, 100)
x = np.array([31,14,2])
y = np.array([90,80,50])
plt.xlabel('Number of Components')
plt.ylabel('Variance Percentage')
plt.title('PCA Components vs Variance')
plt.plot(x, y)
plt.show()

i = 0
plt.ylim(-0.5,1.5)
plt.xlim(-10,650)
plt.plot(Nvalues)
plt.xlabel('Eigen Values')
plt.ylabel('Normalized Vector Value')
plt.title('Normalized Vectors')
plt.show()

#Part C
ui = diff_matrixmean*eigenVectors
print(ui.shape)

uil2norm = linalg.norm(ui,None,axis=0)
print(uil2norm.shape)

normalizedui = ui/uil2norm
print('normalizedui.shape')
print(normalizedui.shape)

#Take First 5 Columns
Img1 = normalizedui[:,623].reshape(30,32)
Img2 = normalizedui[:,622].reshape(30,32)
Img3 = normalizedui[:,621].reshape(30,32)
Img4 = normalizedui[:,620].reshape(30,32)
Img5 = normalizedui[:,619].reshape(30,32)

#Poor Resolution
scipy.misc.imsave('Img1.jpg', Img1)
scipy.misc.imsave('Img2.jpg', Img2)
scipy.misc.imsave('Img3.jpg', Img3)
scipy.misc.imsave('Img4.jpg', Img4)
scipy.misc.imsave('Img5.jpg', Img5)

fig = plt.figure()
a1=fig.add_subplot(321)
imgplot = plt.imshow(Img1, cmap = plt.get_cmap('gray'))
a1.title.set_text("623")
a2=fig.add_subplot(322)
imgplot = plt.imshow(Img2, cmap = plt.get_cmap('gray'))
a2.title.set_text("622")
a3=fig.add_subplot(323)
imgplot = plt.imshow(Img3, cmap = plt.get_cmap('gray'))
a3.title.set_text("621")
a4=fig.add_subplot(324)
imgplot = plt.imshow(Img4, cmap = plt.get_cmap('gray'))
a4.title.set_text("620")
fig.tight_layout() 
plt.show()

#Merge and Save

#Take Last 5 Columns
Img6 = normalizedui[:,0].reshape(30,32)
Img7 = normalizedui[:,1].reshape(30,32)
Img8 = normalizedui[:,2].reshape(30,32)
Img9 = normalizedui[:,3].reshape(30,32)
Img10 = normalizedui[:,4].reshape(30,32)

#Better Resolution
scipy.misc.imsave('Img6.jpg', Img6)
scipy.misc.imsave('Img7.jpg', Img7)
scipy.misc.imsave('Img8.jpg', Img8)
scipy.misc.imsave('Img9.jpg', Img9)
scipy.misc.imsave('Img10.jpg', Img10)


fig = plt.figure()
a1=fig.add_subplot(321)
imgplot = plt.imshow(Img6, cmap = plt.get_cmap('gray'))
a1.title.set_text("0")
a2=fig.add_subplot(322)
imgplot = plt.imshow(Img7, cmap = plt.get_cmap('gray'))
a2.title.set_text("1")
a3=fig.add_subplot(323)
imgplot = plt.imshow(Img8, cmap = plt.get_cmap('gray'))
a3.title.set_text("2")
a4=fig.add_subplot(324)
imgplot = plt.imshow(Img9, cmap = plt.get_cmap('gray'))
a4.title.set_text("3")
fig.tight_layout() 
plt.show()

Tnormalizedui = np.transpose(normalizedui)

#Merge and Save
weights = Tnormalizedui*diff_matrixmean[:,75]
print('weights shape')
print(weights.shape)
print('weights shape')
print(weights[0,0])
print(weights[1,0])
print(weights[2,0])
print(weights[3,0])

comp1 = normalizedui[:,0]*weights[0,0]
comp2 = normalizedui[:,1]*weights[1,0]
comp3 = normalizedui[:,2]*weights[2,0]
comp4 = normalizedui[:,3]*weights[3,0]
comp5 = weights[4,0]*normalizedui[:,4]

print(comp1.shape)

comp100 = normalizedui[:,0]*weights[0,0]

i = 0
for i in range(1,100):
    comp100 = comp100 + normalizedui[:,i]*weights[i,0]

i = 0
comp50 = normalizedui[:,0]*weights[0,0]
for i in range(1,50):
    comp50 = comp50 + normalizedui[:,i]*weights[i,0]

i = 0
comp30 = normalizedui[:,0]*weights[0,0]
for i in range(1,30):
    comp30 = comp30 + normalizedui[:,i]*weights[i,0]

i = 0
comp20 = normalizedui[:,0]*weights[0,0]
for i in range(1,20):
    comp20 = comp20 + normalizedui[:,i]*weights[i,0]

c1 = comp1

ca100 = comp100.reshape(30,32)
ca50 = comp50.reshape(30,32)
ca30 = comp30.reshape(30,32)
ca20 = comp20.reshape(30,32)
ca1 = c1.reshape(30,32)

orgImage = data_matrix[:,75].reshape(30,32)
scipy.misc.imsave('ca100.jpg', ca100)
scipy.misc.imsave('ca50.jpg', ca50)
scipy.misc.imsave('ca30.jpg', ca30)
scipy.misc.imsave('ca20.jpg', ca20)
scipy.misc.imsave('ca1.jpg', ca1)
scipy.misc.imsave('orgImage.jpg', orgImage)

fig = plt.figure()
a1=fig.add_subplot(321)
imgplot = plt.imshow(ca100, cmap = plt.get_cmap('gray'))
a1.title.set_text("100 Components")
a2=fig.add_subplot(322)
imgplot = plt.imshow(ca50, cmap = plt.get_cmap('gray'))
a2.title.set_text("50 Components")
a3=fig.add_subplot(323)
imgplot = plt.imshow(ca30, cmap = plt.get_cmap('gray'))
a3.title.set_text("30 Components")
a4=fig.add_subplot(324)
imgplot = plt.imshow(ca20, cmap = plt.get_cmap('gray'))
a4.title.set_text("20 Components")
fig.tight_layout() 
plt.show()
