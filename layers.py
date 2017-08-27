import tensorflow as tf
import numpy as np


class ConvolutionalLayer:
    def __init__(self, shape, namescope):
        self.namescope = namescope
        with tf.name_scope(self.namescope):
            self.W = weight_variable(shape)
            self.b = bias_variable([shape[-1]])

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def forward(self, Z):
        return tf.nn.relu(self.conv2d(Z, self.W) + self.b)



class PoolingLayer:
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def max_pool_2x2(self, Z):
        return tf.nn.max_pool(Z, ksize=self.ksize, 
                strides=self.strides, padding=self.padding)

    def forward(self, Z):
        return self.max_pool_2x2(Z)


class ConPoolLayer:
    def __init__(self, size, name):
        self.convolution_op = ConvolutionalLayer(size, name)
        self.pool_op = PoolingLayer()
        
    def forward(self, Z):
        h_conv = self.convolution_op.forward(Z)
        h_pool = self.pool_op.forward(h_conv)
        return h_pool


def weight_variable(shape, stddev=.1):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(init, name="W")


def bias_variable(shape, val=.1):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init_biases(shape), name="b")



