import os
import sys
import numpy as np

cwd = os.getcwd()
sys.path.append(cwd + '/../common/')
import data_controller as dc

DATA_FILE_NAME = 'Iris.csv'
IRIS_SETOSA = [0, 'Iris-setosa']
IRIS_VERSICOLOR = [1, 'Iris-versicolor']
IRIS_VIRGINICA = [2, 'Iris-virginica']

class Neural_Network(object):
	def __init__(self, inputLayerSize, outputLayerSize, hiddenLayerSize, numHiddenLayers=1, eta=.2):
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.hiddenLayerSize = hiddenLayerSize
		self.eta = eta

		# weight matrices -- would need more of thiese with more hidden layers
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	def feedForward(self, X):
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	'''
		Derivative of the sigmoid function
	'''
	def dsigmoid(self, z):
		#return np.exp(-z)/((1+np.exp(-z))**2)
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def costFunction(self, X, y):
		self.yHat = self.feedForward(X)
		# Error function E = .5 sum(y - yhat)^2 == sum of squares
		J = 0.5*sum((y-self.yHat)**2)
		return J

	'''
		Computes the derivitate with respect to our weights.

		y - actuals
		X - example 
	'''
	def costFunctionPrime(self, X, y):
		self.yHat = self.feedForward(X)

		# To make a deeper network, stack more of these on top of each other
		# That is, would need delta4/5/6/... where every layer except the last layer
		# (layer that goes to the output) takes the form of the second function (computing
		# delta2). This function is delta_j where j is the layer in question, the last hidden 
		# layer -> output is delta_k and has a different function. In the second calculation
		# the delta2 would be replaced with the delta from the previous layer rather than the
		# delta from the output layer. 

		# derivative of the error function (sum of squares) w.r.t. weights from 
		# layer j to output layer k (the second set of weights)
		delta3 = np.multiply(-(y-self.yHat), self.dsigmoid(self.z3))
		self.djdw2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.dsigmoid(self.z2)
		self.djdw1 = np.dot(X.T, delta2)

		return self.djdw1, self.djdw2

	def adjust_weights(self):
		self.W1 = self.W1 - self.eta*self.djdw1
		self.W2 = self.W2 - self.eta*self.djdw2




if __name__ == "__main__":
	data = dc.get_data(DATA_FILE_NAME)
	NN = Neural_Network(4, 3, 5)
	testSize = 50

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

	rows = []
	for k in train:
		row = train[k]
		rows.append(row)
	# create [samples x input] numpy array, one row for each training data
	trainArray = np.array([row for row in rows])

	rows = []
	for k in test:
		row = test[k]
		rows.append(row)
	testArray = np.array([row for row in rows])

	t = []
	for v in y[:-testSize]:
		# value for each class
		x = [0, 0, 0]
		# actual class has a 100% probability of being that class (1)
		x[v] = 1
		t.append(x)
	trainY = np.array([x for x in t])

	epsilon = .000000000001
	cost1, cost2 = np.array([100]), np.array([0])
	#while cost1 - cost2 > epsilon:
	while sum(cost1-cost2) > epsilon:
		cost1 = NN.costFunction(trainArray[1:5], trainY[1:5])
		# 1:2 - must pass costFunctionPrime a list of lists
		djdw1, djdw2 = NN.costFunctionPrime(trainArray[1:5], trainY[1:5])
		NN.adjust_weights()
		cost2 = NN.costFunction(trainArray[1:5], trainY[1:5])
		print(sum(cost1-cost2))

	# may not be right
	# all testyHats are the practically same...
	print('\n')
	testYHat = NN.feedForward(testArray)
		
	t = []
	for v in y[-testSize:]:
		x = [0, 0, 0]
		x[v] = 1
		t.append(x)
	testY = np.array([x for x in t])
	testY = y[-testSize:]

	right = 0
	total = 0
	yHats = []
	for i in range(0, len(testYHat)):
		yHats.append(testYHat[i].tolist().index(max(testYHat[i])))

	for i in range(0, len(testY)):
		if testY[i] == yHats[i]:
			right = right + 1
		total = total+1

	print(yHats, '\n\n', testY)
	print('Accuracy: ' + str(right/total))