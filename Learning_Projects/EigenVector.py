# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:44:14 2016

@author: sannadurai
"""

import numpy as np
from sympy import Matrix
from scipy import linalg as la
import sys
print (sys.version)

A = Matrix([[-2,2],[-6,5]])
print(A.eigenvals())

#Create an nn array with checkerboard pattern of zeros and ones.
Z = np.tile(np.array([[0,1],[1,0]]),(4,4))
print(Z)

