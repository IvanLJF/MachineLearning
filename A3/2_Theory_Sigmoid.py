import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-70, 70,1)
sig = sigmoid(x)
plt.plot(x,sig)
plt.title('Sigmoid Weight 1')
plt.show()

x = np.arange(-70, 70,5)
sig = sigmoid(x)
plt.title('Sigmoid Weight 5')
plt.plot(x,sig)
plt.show()

x = np.arange(-70,70,100)
sig = sigmoid(x)
plt.title('Sigmoid Weight 100')
plt.plot(x,sig)
plt.show()