""" Class containing the class definition of an artificial neural network.

This file contains the class definition of an artificial neural network in python.

TODO:
    All
"""

import numpy as np


class Ann(object):
    @staticmethod
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_logistic(x):
        return Ann.logistic(x) * (1 - Ann.logistic(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1.0 - np.tanh(x) ** 2

    @staticmethod
    def loss(y, y_hat):
        return 0.5 * sum((y - y_hat)**2)

    def __init__(self, number_hidden_layers, hidden_layer_sizes, output_layer_size, input_layer_size,
                 activation='logistic', l_rate=0.1):
        """ Creates an artificial neural network with the parameters provided.

        Args:
            number_hidden_layers (int): the number of hidden layers in the network
            hidden_layer_sizes (list): list of integers corresponding to the sizes (number of nodes) of each hidden layer
            output_layer_size (int): number of nodes in the output layer
            input_layer_size (int): the length of the feature vector; number of inputs to the network
            activation (string): default = logistic; which activation function to use { tanh, logistic }
            l_rate (float): the learning rate of the algorithm, typically denoted eta
        """
        self.hidden_layer_count = number_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_classes = output_layer_size
        self.n_inputs = input_layer_size
        self.hidden_layers = []
        self.output_layer = None
        self.activation = activation
        self.eta = l_rate
        self.residuals = []
        self.x = None

        if activation == 'logistic':
            self.activation_function = Ann.logistic
            self.activation_derivative = Ann.d_logistic
        else:
            self.activation_function = Ann.tanh
            self.activation_derivative = Ann.d_tanh

        if len(self.hidden_layer_sizes) != self.hidden_layer_count:
            print("\nFailed to create neural network. Hidden layer count is not equal to number of hidden layers sizes"
                  " provided.")
            exit(0)

        self.build_model()

    def build_model(self):
        """ Builds the TensorFlow model of the neural network
        """
        self.hidden_layers.append({'weights': np.random.randn(self.n_inputs, self.hidden_layer_sizes[0]),
                                   'biases': np.random.randn(self.hidden_layer_sizes[0]),
                                   'output': None,
                                   'input': None,
                                   'deltaW': 0.0,
                                   'deltaB': 0.0})
        for i in range(1, self.hidden_layer_count):
            self.hidden_layers.append({'weights': np.random.randn(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i]),
                                       'biases': np.random.randn(self.hidden_layer_sizes[i]),
                                       'output': None,
                                       'input': None,
                                       'deltaW': 0.0,
                                       'deltaB': 0.0})
        self.output_layer = {'weights': np.random.randn(self.hidden_layer_sizes[self.hidden_layer_count - 1], self.n_classes),
                             'biases': np.random.randn(self.n_classes),
                             'output': None,
                             'input': None,
                             'deltaW': 0.0,
                             'deltaB': 0.0}

    def feed_forward(self, x):
        """ Runs an example vector through the NN and produces a yHat response
        """
        self.x = x
        z_mat = np.add(np.matmul(x, self.hidden_layers[0]['weights']), self.hidden_layers[0]['biases'])
        self.hidden_layers[0]['input'] = z_mat
        z_mat = self.activation_function(z_mat)    # This is the activation function, can select another or make it a parameter
        self.hidden_layers[0]['output'] = z_mat
        for i in range(1, self.hidden_layer_count):
            z_mat = np.add(np.matmul(z_mat, self.hidden_layers[i]['weights']), self.hidden_layers[i]['biases'])
            self.hidden_layers[i]['input'] = z_mat
            z_mat = self.activation_function(z_mat)
            self.hidden_layers[i]['output'] = z_mat

        self.output_layer['input'] = z_mat
        output = np.matmul(z_mat, self.output_layer['weights'] + self.output_layer['biases'])
        self.output_layer['output'] = output
        return output

    def back_propagate(self, y, y_hat):
        self.compute_gradients(y, y_hat)
        self.update_weights()

    def compute_gradients(self, y, y_hat):
        dk = (y - y_hat)
        self.output_layer['deltaB'] = dk
        dk = np.multiply(dk, self.activation_derivative(self.output_layer['output']))
        self.output_layer['deltaW'] = np.dot(self.output_layer['input'].transpose(), dk)
        for i in reversed(range(0, self.hidden_layer_count)):
            dk = np.dot(dk, self.hidden_layers[i]['weights'].transpose())
            self.hidden_layers[i]['deltaB'] = dk
            dk *= self.activation_derivative(self.hidden_layers[i]['input'])
            if i == 0:
                self.hidden_layers[i]['deltaW'] = np.dot(self.x.transpose(), dk)
            else:
                self.hidden_layers[i]['deltaW'] = np.dot(self.hidden_layers[i - 1]['output'].transpose(), dk)

    def update_weights(self):
        return None


    def predict(self, x):
        """ Returns a prediction for the provided feature vector
        """
        return self.feed_forward(x)
