"""Deep Convolutional GAN
"""
import tensorflow as tf
from util import get_pickled_data
import matplotlib.pyplot as plt
import numpy as np
from ann import ANN 

class ConvolutionalLayer:
    def __init__(self, layer_info):
        self.namescope = layer_info["namescope"]

        with tf.name_scope(self.namescope):
            # Try to load pretrained weights, initialize random weights if 
            # pretrained weights are not included
            self.W = weight_variable(shape)
            self.b = bias_variable([shape[-1]])

    def forward(self, Z):
        """Propagate features through Convolutional layer
        """
        with tf.name_scope(self.namescope):
            h_conv = tf.nn.relu(
                        tf.nn.conv2d(
                            Z, self.W, strides=[1, 1, 1, 1], padding='SAME') 
                        + self.b)

        return h_conv


class DeconvolutioinalLayer:
    """Fractionally strided convolution layer
    """
    pass


class DCGAN:

    def __init__(self):

        with tf.name_scope("Discriminator"):
            self._init_discriminator()

        with tf.name_scope("Generator"):
            self._init_generator()

    def _init_discriminator():
        pass

    def _init_generator():
        pass



def weight_variable(shape, stddev=.1):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(init, name="W")

def bias_variable(shape, val=.1):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init, name="b")

if __name__ == "__main__":
    plt.ion()

    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D = Xtrain.shape

    generator_params = {
            "z": 100,
            "projection": 128,
            convolution_layers:[
                {"depth": , "strides":2, "window":5},
                {"depth":1, "strides":2, "window":5},
                ],
            fullyconnected_layers:[1024],
            "activation":tf.nn.sigmoid
            }

    descriminator_params = {
            convolution_layers:[
                {"depth":2, "strides":2, "window":5},
                {"depth":64, "strides":2, "window":5},
                ],
            fullyconnected_layers:[1024],
            }
    
    with tf.Session() as session:
        dcgan = DCGAN()
        session.run(tf.global_variables_initializer())
        dcgan.set_session(session)
        dcgan.fit(Xtrain)

