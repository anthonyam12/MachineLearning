import numpy as np
import pandas as pd
import os 
import sys 
from ann import * 

cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
from trainer import *

trainSize = 150
df = pd.read_csv('world_happiness2016.csv')
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

NN = NeuralNetwork(7, 1, 11)
T = trainer(NN)
T.train(trainX, trainY)

yHat = NN.forward(testX)
print(yHat)
print(testY)