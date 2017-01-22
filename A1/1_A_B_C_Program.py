import numpy as np
import sys
#2.7.12 |Anaconda 4.0.0 (64-bit)

#Get version of python
print (sys.version)

#(a) Create an nxn array with checkerboard pattern of zeros and ones.
#Declare and initialize value of n
n = 15
Z = np.zeros((n,n),dtype=int)
Z[1::2,::2]=1
Z[::2,1::2]=1
print('Checker board pattern')
print(Z)

#(b) Given an n x n array, sort the rows of array according to mth column of array
n = 5
m = 2
Z = np.random.randint(0,10,(n, n))
sortedarray = Z[Z[:,m].argsort()]
print('Original Input')
print(Z)
print('Sorted based on element position 2')
print(sortedarray)

#(c) Create an nxn array with (i+j)th-entry equal to i+j.
n = 3
a = np.repeat(np.arange(n), n)
# C-like index order
i = np.reshape((a),(n,n),order="C")
#Fortran-like index order
j = np.reshape((a),(n,n),order="F")
k = i+j
print('nxn array with (i+j)th-entry equal to i+j')
print(k)
