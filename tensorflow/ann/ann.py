""" Class containing the class definition of an artificial neural network.

This file contains the class definition of an artificial neural network in python.
The purpose of this class is to allow the creation of GENERAL neural networks rather than
limiting the size.

"""
# import the TensorFlow library
import tensorflow as tf
import numpy as np


class ANN(object):
    # Static methods for the activation functions and activation derivatives.
    @staticmethod
    def logistic(x):
        return 1 / (1 + tf.exp(-x))

    @staticmethod
    def d_logistic(x):
        return ANN.logistic(x) * (1 - ANN.logistic(x))

    @staticmethod
    def tanh(x):
        return tf.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1.0 - tf.tanh(x) ** 2

    @staticmethod
    def loss(y, y_hat):
        return 0.5 * sum((y - y_hat)**2)

    def __init__(self, number_hidden_layers, hidden_layer_sizes, output_layer_size, input_layer_size,
                 activation='logistic'):
        """ Creates an artificial neural network with the parameters provided.

        Args:
            number_hidden_layers (int): the number of hidden layers in the network
            hidden_layer_sizes (list): list of integers corresponding to the sizes (number of nodes) of each hidden layer
            output_layer_size (int): number of nodes in the output layer
            input_layer_size (int): the length of the feature vector; number of inputs to the network
            activation (string): default = logistic; which activation function to use { tanh, logistic }
        """

        # Set the parameters of the network
        self.hidden_layer_count = number_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_classes = output_layer_size
        self.n_inputs = input_layer_size
        self.hidden_layers = []
        self.output_layer = None
        self.activation = activation
        self.x = None

        if activation == 'logistic':
            self.activation_function = ANN.logistic
            self.activation_derivative = ANN.d_logistic
        else:
            self.activation_function = ANN.tanh
            self.activation_derivative = ANN.d_tanh

        # invalid parameter settings
        if len(self.hidden_layer_sizes) != self.hidden_layer_count:
            print("\nFailed to create neural network. Hidden layer count is not equal to number of hidden layers sizes"
                  " provided.")
            exit(0)

        # creates the Neural Network, builds the model
        self.build_model()

    def build_model(self):
        """ Builds the TensorFlow model of the neural network with the provided parameters.
        """

        # The first hidden layer has a different size than the others since the inputs are the x vector values
        self.hidden_layers.append({'weights': tf.Variable(tf.random_normal([self.n_inputs, self.hidden_layer_sizes[0]], dtype=tf.float64)),
                                   'biases': tf.Variable(tf.random_normal([self.hidden_layer_sizes[0]], dtype=tf.float64))})

        # create the hidden layers based on the sizes in self.hidden_layer_sizes
        for i in range(1, self.hidden_layer_count):
            self.hidden_layers.append({'weights': tf.Variable(tf.random_normal(
                                                [self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i]],
                                                dtype=tf.float64)),
                                       'biases': tf.Variable(tf.random_normal([self.hidden_layer_sizes[i]], dtype=tf.float64))})

        # create the output layer
        self.output_layer = {'weights': tf.Variable(tf.random_normal(
                                        [self.hidden_layer_sizes[self.hidden_layer_count - 1], self.n_classes],
                                        dtype=tf.float64)),
                             'biases': tf.Variable(tf.random_normal([self.n_classes], dtype=tf.float64))}

    def feed_forward(self, x):
        """ Runs an example vector through the NN and produces a yHat response
        """

        # first step uses x vector as input)
        il = tf.add(tf.matmul(x, self.hidden_layers[0]['weights']), self.hidden_layers[0]['biases'])
        a = self.activation_function(il)

        # feed the outputs forward through the hidden layers
        for i in range(1, self.hidden_layer_count):
            l = tf.add(tf.matmul(a, self.hidden_layers[i]['weights']), self.hidden_layers[i]['biases'])
            a = self.activation_function(l)

        # get the output of the network
        output = tf.matmul(a, self.output_layer['weights']) + self.output_layer['biases']

        # optionally, could send the output through an activation function or THE activation function of the NN
        # output = self.activation_function(output)
        # OR
        # output = tf.nn.relu(output)
        # ...

        return output

    def predict(self, x):
        """ Returns a prediction for the provided feature vector
        """
        return self.feed_forward(x)

    def train(self, trainx, trainy, number_epochs=50):
        """ Trains the neural network using trainx and trainy data
        :param trainx: x data to train neural network
        :param trainy: y data used to train neural network
        :param number_epochs: default = 50, number of training epochs
        :return:
        """
        # number of columns
        x = tf.placeholder('float64', [None, trainx.shape[1]])
        y = tf.placeholder('float64', [None, trainy.shape[1]])

        y_hat = self.predict(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)  # https://arxiv.org/pdf/1412.6980.pdf
        with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                for epoch in range(number_epochs):
                    loss = 0
                    # assumes batch size = length of trainX, could change this by passing in batch size and partitioning
                    # the datasets.
                    _, c = session.run(optimizer, feed_dict={x: np.atleast_2d(trainx[1, :]), y: np.atleast_2d(trainy[1, :])})
                    print('Epoch:', epoch, 'of', number_epochs, '\tloss:', c)
