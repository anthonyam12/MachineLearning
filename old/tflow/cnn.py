import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

hidden_layer_nodes = [500, 500, 500, 500, 500]
classes = 10
batch_size = 100
input_layer_size = 784 # num pixels (28x28)

# input layer, { hidden layers }, output layer
layer_sizes = [input_layer_size]
for size in hidden_layer_nodes:
    layer_sizes.append(size)
layer_sizes.append(classes)

x = tf.placeholder('float', [None, input_layer_size])
y = tf.placeholder('float')

def train(x, layer_sizes):
    prediction = neural_network_model(x, layer_sizes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # AdamOptimizer(...) takes optional learning rate param, default = .001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = 75
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch ', e, ' completed out of ', epochs, ' loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
    
def neural_network_model(data, layer_sizes):
    # { hidden layer(s) }, output layer
    layers = [data]
    for i in range(1, len(layer_sizes)):
        layers.append(
                {'weights':tf.Variable(tf.random_normal([layer_sizes[i-1], layer_sizes[i]])),
                 'biases':tf.Variable(tf.random_normal([layer_sizes[i]]))}
            )
    # relu = Re(ctified) L(inear) U(nit) is an activation function that makes
    # the input = 0 if the input is negative
    # sigmoid function = tf.sigmoid(x) or tf.nn.sigmoid(x)
    result = tf.add(tf.matmul(layers[0], layers[1]['weights']), layers[1]['biases'])
    result = tf.nn.relu(result)
    for i in range(2, len(layers)-1):
        result = tf.add(tf.matmul(result, layers[i]['weights']), layers[i]['biases'])
        result = tf.nn.relu(result)
    result = tf.matmul(result, layers[len(layers)-1]['weights']) + layers[len(layers)-1]['biases']
    return result
    
train(x, layer_sizes)
