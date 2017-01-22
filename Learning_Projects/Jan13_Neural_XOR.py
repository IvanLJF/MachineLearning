#Runs in Python 2.7.12
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#XOR Operation
#   0,0 -0
#   0,1 - 1
#   1,0 - 1
#   1,1 - 0

#2 - 2 Neurons Input
#3 - HiddenLayer
#1 - Output layer
neuralNetwork = buildNetwork(2,3,1)

#2 Dimension input, 1 Dimension Output
dataSet = SupervisedDataSet(2,1)

#Add Training Data
#x,y - 2D
#Output - 1D
dataSet.addSample((0,0),(0,))
dataSet.addSample((1,0),(1,))
dataSet.addSample((0,1),(1,))
dataSet.addSample((1,1),(0,))

#Trainer Backpropagation
trainer = BackpropTrainer(neuralNetwork, dataSet)

for i in range(1,10000):
    trainer.train()
    
    if(i%4000==0):
        print('Iteration - ',i)
        print(neuralNetwork.activate([0,0]))
        print(neuralNetwork.activate([1,0]))
        print(neuralNetwork.activate([0,1]))
        print(neuralNetwork.activate([1,1]))
        

#Output
#('Iteration - ', 4000)
#[ 0.07299632]
#[ 0.92082515]
#[ 0.95759942]
#[ 0.04306811]
#('Iteration - ', 8000)
#[ 0.00045084]
#[ 0.99952651]
#[ 0.99974768]
#[ 0.00023939]


neuralNetwork = buildNetwork(2,6,1)

#Trainer Backpropagation
trainer = BackpropTrainer(neuralNetwork, dataSet)

for i in range(1,10000):
    trainer.train()
    
    if(i%4000==0):
        print('Iteration - ',i)
        print(neuralNetwork.activate([0,0]))
        print(neuralNetwork.activate([1,0]))
        print(neuralNetwork.activate([0,1]))
        print(neuralNetwork.activate([1,1]))

#Output
#('Iteration - ', 4000)
#[ 0.07809688]
#[ 0.85013337]
#[ 0.89882611]
#[ 0.15568355]
#('Iteration - ', 8000)
#[ 0.0005494]
#[ 0.9988819]
#[ 0.99922662]
#[ 0.00120211]
#With more hidden layers convergence rate is faster
