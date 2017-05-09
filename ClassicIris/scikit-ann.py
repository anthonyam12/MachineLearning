from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
import data_controller as dc

DATA_FILE_NAME = 'Iris.csv'
IRIS_SETOSA = [0, 'Iris-setosa']
IRIS_VERSICOLOR = [1, 'Iris-versicolor']
IRIS_VIRGINICA = [2, 'Iris-virginica']

data = dc.get_data(DATA_FILE_NAME)
testSize = 100

y = []
for key in data:
	row=data[key]
	if row[0] in IRIS_SETOSA:
		y.append(IRIS_SETOSA[0])
	elif row[0] in IRIS_VIRGINICA:
		y.append(IRIS_VIRGINICA[0])
	elif row[0] in IRIS_VERSICOLOR:
		y.append(IRIS_VERSICOLOR[0])

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

trainY = y[:-testSize]

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

testY = y[-testSize:]

x=[[0, 0], [1., 1.]]
y = [0, 1]

clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)

clf.fit(trainX, trainY)
print(clf.score(testX, testY))
yHats = clf.predict(testX)

print(np.round(yHats))
print(testY)