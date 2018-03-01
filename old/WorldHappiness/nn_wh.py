import numpy as np
import pandas as pd
import os 
import sys 

cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
from trainer import *
from ann import * 


trainSize = 150
df = pd.read_csv('world_happiness2016_randCol.csv')
df = df.reindex(np.random.permutation(df.index))

y = df.iloc[:,3].as_matrix()
x = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1).as_matrix()

testX = x.tolist()[trainSize:]
testY = y.tolist()[trainSize:]
trainX = x.tolist()[:trainSize]
trainY = y.tolist()[:trainSize]

trainX = np.array([i for i in trainX])
testX = np.array([i for i in testX])
trainY = np.array([[i] for i in trainY])
testY = np.array([[i] for i in testY])

<<<<<<< HEAD:WorldHappiness/nn_wh.py
NN = NeuralNetwork(8, 1, 11)
T = trainer(NN)
T.train(trainX, trainY)

yHat = NN.forward(testX)
NN.printWeights()
print(yHat)
print(testY)
=======
NN = NeuralNetwork(7, 1, 11)

#yHat = NN.forward(trainX[0])

NN.costPrime(trainX[0], trainY[0])

#T = trainer(NN)
#T.train(trainX, trainY)

#print(NN.errors)

#yHat = NN.forward(testX)
# print(yHat)
# print(testY)
>>>>>>> 1f4b0c00b3626afe025707d34e0a656a63fd6075:old/WorldHappiness/nn_wh.py
