"""Implementation of the VGG16 network 

** Because this is benchmarked to MNIST, the first 10 conpool layers are omitted
and the layer depths are reduced by a factor of 3 (only one color channel)

Implementation of this is intentionally verbose so that each operation is clear

**incomplete**
"""
import tensorflow as tf
import numpy as np

from util import y2indicator, accuracy, get_pickled_image_data
from ann import ANN 


class ConvolutionalLayer:
    def __init__(self, layer_info):
        self.namescope = layer_info["namescope"]

        with tf.name_scope(self.namescope):
            # Try to load pretrained weights, initialize random weights if 
            # pretrained weights are not included
            try:
                self.W = layer_info["layer_weights"]
                self.b = layer_info["layer_bias"]
            else:
                print("Could not load pretraind weights, initializing random weights...")
                assert False

    def forward(self, Z):
        """Propagate features through Convolutional layer
        """
        with tf.name_scope(self.namescope):
            h_conv = tf.nn.relu(
                        tf.nn.conv2d(
                            Z, self.W, strides=[1, 1, 1, 1], padding='SAME') 
                        + self.b)

        return h_conv


class PoolingLayer:
    def __init__(self):
        self.namescope = layer_info["namescope"]
        self.ksize = layer_info["ksize"]
        self.pooling_strides = layer_info["pooling_strides"]

    def forward(self, h_conv):
        """Propagate features through Convolutional layer
        """

        with tf.name_scope(self.namescope):
            h_pool = tf.nn.max_pool(h_conv, ksize=self.ksize, 
                strides=self.pooling_strides, padding='SAME')

        return h_pool


class VGG16:
    def __init__(self):
        self.Xin = tf.placeholder(
                tf.float32, 
                [None, *input_size], 
                name="X")
        self.labels = tf.placeholder(
                tf.float32, 
                [None, output_size], 
                name="labels")


        self.conpool_layers = []
        self.fullcon_layers = []

        """First 10 stages omitted for MNIST
        """
        ## Conpool Stage 1
        # con1_1
        # con1_2
        # pool1 
        self.conpool_layers.extend([
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                PoolingLayer()
                ])

        ## Conpool Stage 2
        # con2_1
        # con2_2
        # pool2 
        self.conpool_layers.extend([
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                PoolingLayer()
                )

        ## Conpool Stage 3
        # con3_1
        # con3_2
        # con3_3
        # pool3 
        self.conpool_layers.extend([
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                PoolingLayer()
                ])

        ## Conpool Stage 4
        # con4_1
        # con4_2
        # con4_3
        # pool4 
        self.conpool_layers.extend([
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                PoolingLayer()
                ])

        ## Conpool Stage 5
        # con5_1
        # con5_2
        # con5_3
        # pool5 
        self.conpool_layers.extend([
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                ConvolutionalLayer(),
                PoolingLayer()
                ])

        ## Fully-connected Stage 6
        # FC1
        # FC2
        # FC3
        self.fullcon_layers.extend([
                ANN(),
                ANN(),
                ANN(),
                ])


def weight_variable(shape, stddev=.1):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(init, name="W")


def bias_variable(shape, val=.1):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init, name="b")


if __name__ == "__main__":
    pass
