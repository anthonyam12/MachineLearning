import tensorflow as tf
import numpy as np

class TensorflowLSTM(object):
    def __init__(self, input_size, hidden_layers, hidden_layer_sizes, output_size,
                 epochs=1000, batch_size=1):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None # in tf, just add each layer to the model
        self.layers = []
        self.output_layer = []
        self.graph = tf.Graph()

        self.create_model()
        self.initial_state = self.model.zero_state(self.batch_size, tf.float32)
        self.state = self.model.zero_state(self.batch_size, tf.float32)

    def create_model(self):
        with self.graph.as_default():
            # add input layer
            self.layers.append(tf.contrib.rnn.BasicLSTMCell(self.input_size, reuse=tf.get_variable_scope().reuse))
            # add hidden layers
            for i in range(0, self.hidden_layers):
                self.layers.append(tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_sizes[i], reuse=tf.get_variable_scope().reuse))
            # add output layer
            # self.output_layer = {'weights':tf.Variable(tf.random_normal([self.hidden_layer_sizes[self.hidden_layers-1], self.output_size])),
            #                      'biases':tf.Variable(tf.random_normal([self.output_size]))}
            self.layers.append(tf.contrib.rnn.BasicLSTMCell(self.output_size, reuse=tf.get_variable_scope().reuse))
            self.model = tf.contrib.rnn.MultiRNNCell(self.layers)


    def predict(self, pred_x):
        # TODO handle single value inputs
        # TODO as it stands can only
        pred_x = pred_x.reshape(-1, len(pred_x), self.input_size)

        with tf.Session(graph = self.graph) as sess:
            init = tf.global_variables_initializer()
            init.run()
            print(sess.run(tf.report_uninitialized_variables()))
            # init_vars_op = tf.initialize_variables()
            output = sess.run(self.output, feed_dict={self.x: pred_x})
        return output
    
    def train(self, trainx, trainy):
        # Finish defining the graph
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, len(trainx), self.input_size])
            y = tf.placeholder(tf.float32, [None, len(trainy), self.output_size])

            trainx = trainx.reshape(-1, len(trainx), self.input_size)
            trainy = trainy.reshape(-1, len(trainy), self.input_size)

            output, self.state = tf.nn.dynamic_rnn(self.model, self.x, dtype=tf.float32)
            output = tf.reshape(output, [-1, self.hidden_layers])
            output = tf.layers.dense(output, self.output_size)
            self.output = tf.reshape(output, [-1, len(trainx), self.output_size])

            loss = tf.reduce_mean(tf.square(self.output-y))
            optimizer = tf.train.AdamOptimizer()
            op = optimizer.minimize(loss)
            init = tf.global_variables_initializer()

        with tf.Session(graph=self.graph) as sess:
            init.run()
            for i in range(self.epochs):
                sess.run(op, feed_dict={self.x: trainx, y:trainy})
                error = loss.eval(feed_dict={self.x:trainx, y:trainy})
                print(i, "\tMSE: ", error);
