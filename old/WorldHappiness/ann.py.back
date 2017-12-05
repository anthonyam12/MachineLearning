import numpy as np
from scipy import optimize

# source https://www.youtube.com/watch?v=9KM9Td6RVgQ&list=PLxtZWOzqbApW6Yvw41wDf0uc5hixr1b02&index=6
class NeuralNetwork(object):
	def __init__(self, inputLayerSize, outputLayerSize, hiddenLayerSize):
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.hiddenLayerSize = hiddenLayerSize

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	def forward(self, X):
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.z3
		return yHat

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def dsigmoid(self, z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	# sum of squared errors
	def cost(self, X, y):
		yHat = self.forward(X)
		E = 0.5*sum((y-yHat)**2)
		return E

	def costPrime(self, X, y):
		self.yHat = self.forward(X)

		deltak = np.multiply(-(y-self.yHat), self.dsigmoid(self.z3))
		dEdWjk = np.dot(self.a2.T, deltak)

		deltaj = np.dot(deltak, self.W2.T)*self.dsigmoid(self.z2)
		dEdWij = np.dot(X.T, deltaj)

		return dEdWjk, dEdWij

	# Numerical Gradient Checking - determine if our NN gradient descent is working
	def getParams(self):
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		dEdWjk, dEdWij = self.costPrime(X, y)
		return np.concatenate((dEdWij.ravel(), dEdWjk.ravel()))