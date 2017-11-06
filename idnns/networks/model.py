import functools
import numpy as np
from idnns.networks.models import multi_layer_perceptron
from idnns.networks.models import deepnn
from idnns.networks.ops import *


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        # print hasattr(self, attribute)
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
    """A class that represent model of network"""

    def __init__(self, input_size, layerSize, num_of_classes, learning_rate_local=0.001, save_file='',
                 activation_function=0, cov_net=False):
        self.covnet = cov_net
        self.input_size = input_size
        self.layerSize = layerSize
        self.all_layer_sizes = np.copy(layerSize)
        self.all_layer_sizes = np.insert(self.all_layer_sizes, 0, input_size)
        self.num_of_classes = num_of_classes
        self._num_of_layers = len(layerSize) + 1
        self.learning_rate_local = learning_rate_local
        self._save_file = save_file
        self.hidden = None
        self.savers = []
        if activation_function == 1:
            self.activation_function = tf.nn.relu
        elif activation_function == 2:
            self.activation_function = None
        else:
            self.activation_function = tf.nn.tanh
        self.prediction
        self.optimize
        self.accuracy

    def initilizae_layer(self, name_scope, row_size, col_size, activation_function, last_hidden):
        # Build layer of the network with weights and biases
        weights = get_scope_variable(name_scope=name_scope, var="weights",
                                     shape=[row_size, col_size],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0 / np.sqrt(
                                         float(row_size))))
        biases = get_scope_variable(name_scope=name_scope, var='biases', shape=[col_size],
                                    initializer=tf.constant_initializer(0.0))

        self.weights_all.append(weights)
        self.biases_all.append(biases)
        variable_summaries(weights)
        variable_summaries(biases)
        # with tf.variable_scope(name_scope) as scope:  # >>> changed per Mike to make TensorBoard cleaner
        input = tf.matmul(last_hidden, weights) + biases
        if activation_function is None:
            output = input
        else:
            output = activation_function(input, name='output')
        self.inputs.append(input)
        self.hidden.append(output)
        return output

    @property
    def num_of_layers(self):
        return self._num_of_layers

    @property
    def hidden_layers(self):
        """The hidden layers of the network"""
        if self.hidden is None:
            self.hidden, self.inputs, self.weights_all, self.biases_all = [], [], [], []
            last_hidden = self.x
            if self.covnet == 1:
                y_conv, self._drouput, self.hidden, self.inputs = deepnn(self.x)
            elif self.covnet == 2:
                y_c, self.hidden, self.inputs = multi_layer_perceptron(self.x, self.input_size, self.num_of_classes,
                                                                       self.layerSize[0], self.layerSize[1])
            else:

                self._drouput = 'dr'
                # self.hidden.append(self.x)
                for i in range(1, len(self.all_layer_sizes)):
                    name_scope = 'hidden' + str(i - 1)
                    row_size, col_size = self.all_layer_sizes[i - 1], self.all_layer_sizes[i]
                    activation_function = self.activation_function
                    last_hidden = self.initilizae_layer(name_scope, row_size, col_size, activation_function,
                                                        last_hidden)
                name_scope = 'final_layer'
                row_size, col_size = self.layerSize[-1], self.num_of_classes
                activation_function = tf.nn.softmax
                last_hidden = self.initilizae_layer(name_scope, row_size, col_size, activation_function, last_hidden)
        return self.hidden

    @lazy_property
    def prediction(self):
        logits = self.hidden_layers[-1]
        return logits

    @lazy_property
    def drouput(self):
        return self._drouput

    @property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_local).minimize(self.cross_entropy)

        return optimizer

    @lazy_property
    def x(self):
        return tf.placeholder(tf.float32, shape=[None, self.input_size], name='x')

    @lazy_property
    def labels(self):
        return tf.placeholder(tf.float32, shape=[None, self.num_of_classes], name='y_true')

    @lazy_property
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    @lazy_property
    def cross_entropy(self):
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.labels * tf.log(tf.clip_by_value(self.prediction, 1e-50, 1.0)), reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)
        return cross_entropy

    @property
    def save_file(self):
        return self._save_file
