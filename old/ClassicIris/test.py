import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt 

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
import data_controller as dc

from trainer import *
from ann import *

DATA_FILE_NAME = 'Iris.csv'
IRIS_SETOSA = [0, 'Iris-setosa']
IRIS_VERSICOLOR = [1, 'Iris-versicolor']
IRIS_VIRGINICA = [2, 'Iris-virginica']

def computeNumericalGradient(N, X, y):
	paramsInitial = N.getParams()
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	e = 1e-4

	for p in range(len(paramsInitial)):
		perturb[p] = e
		N.setParams(paramsInitial + perturb)
		loss2 = N.cost(X, y)

		N.setParams(paramsInitial - perturb)
		loss1 = N.cost(X, y)

		numgrad[p] = (loss2 - loss1) / (2*e)
		perturb[p] = 0

	N.setParams(paramsInitial)
	return numgrad

# x = np.array(([3,5], [5,1], [10,2]), dtype=float)
# y = np.array(([75], [82], [93]), dtype=float)

# # scale the data
# x = x/np.max(x, axis=0)
# y = y/100 # test scores in this case so max is 100
# eta = 3

data = dc.get_data(DATA_FILE_NAME)
NN = NeuralNetwork(4, 1, 5)
testSize = 100

y = []
for key in data:
	row=data[key]
	if row[0] in IRIS_SETOSA:
		y.append(IRIS_SETOSA[0]/2)
	elif row[0] in IRIS_VIRGINICA:
		y.append(IRIS_VIRGINICA[0]/2)
	elif row[0] in IRIS_VERSICOLOR:
		y.append(IRIS_VERSICOLOR[0]/2)

	data[key] = data[key][1:]
	data[key] = data[key][:-1]
	data[key][0] = float(data[key][0])
	data[key][1] = float(data[key][1])
	data[key][2] = float(data[key][2])
	data[key][3] = float(data[key][3])

train = dict()
for k, v in list(data.items())[:-testSize]:
	train[k] = v

test = dict()
for k, v in list(data.items())[-testSize:]:
	test[k] = v

dy = y[:-testSize]
trainY = np.array([[i] for i in dy])

rows = []
for k in train:
	row = train[k]
	rows.append(row)
# create [samples x input] numpy array, one row for each training data
trainX = np.array([row for row in rows])
trainX = trainX/np.max(trainX, axis=0)

rows = []
for k in test:
	row = test[k]
	rows.append(row)
# create [samples x input] numpy array, one row for each training data
testX = np.array([row for row in rows])
testX = testX/np.max(testX, axis=0)
 
dy = y[-testSize:]
testY = np.array([[i] for i in dy])

print(trainX)
input()

T = trainer(NN)
T.train(trainX, trainY)

# plt.plot(T.E)
# plt.grid(1)
# plt.xlabel('Iterations')
# plt.ylabel('Cost')
# plt.show()

yHats = np.round(2*NN.forward(testX).T)
y = (2*testY).T

errorDiff = (yHats-y)**2
print('\n\nMisclassified ' + str(np.sum(errorDiff)) + ' out of ' + str(testSize))