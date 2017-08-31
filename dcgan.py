"""Deep Convolutional GAN
"""
import tensorflow as tf
from util import get_pickled_image_data
import matplotlib.pyplot as plt
import numpy as np
from ann import ANN 

class FullyConnectedLayer:
    def __init__(self, shape, activation_fn=tf.nn.sigmoid):
        """
        Args
        ----
        shape (list, ints) : [input_size, output_size]
        namescope (str) : name scope of the variables
        activation_fn : the activation function to use 
        """
        self.fn = activation_fn

        self.W = weight_variable(shape)
        self.b = bias_variable([shape[-1]])

    def forward(self, Z):
        """Propagate data through layer
        """
        return self.fn(tf.matmul(Z, self.W) + self.b)

    def forward_without_activation(self, Z):
        """Propagate data through the layer but do not apply a nonlinear 
        activation function. 

        Used for the final output layer
        """
        return tf.matmul(Z, self.W) + self.b


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
    def __init__(self, mi, mo, output_shape, apply_batch_norm, 
            window, stride, activation):

        pass

    def forward(self, Z):
        return Z


class DCGAN:

    BATCH_SIZE = 100

    def __init__(self, discriminator_params, generator_params, image_size=28):

        self.image_size = image_size

        with tf.name_scope("Discriminator"):
            self._init_discriminator(discriminator_params)

        with tf.name_scope("Generator"):
            self._init_generator(generator_params)

    def _init_discriminator(self, params):
        pass

    def _init_generator(self, params):
        """
        """

        dims = [self.image_size]
        for layer_params in reversed(params["convolution_layers"]):
            stride = layer_params["stride"]
            dim = np.ceil(dims[-1] // stride).astype(int)
            dims.append(dim)

        dims = list(reversed(dims))

        # initial input into the generator
        mi = params["latent_vars"]
        self.g_latent_vars = tf.random_uniform(
                [self.BATCH_SIZE, mi],
                minval=0, maxval=1,
                name="latent_variables"
                )

        self.g_dense_layers = []
        for mo in params["fullyconnected_layers"]:
            self.g_dense_layers.append(
                    FullyConnectedLayer(shape=(mi, mo))
                    )

        self.g_convolutional_layers = []
        for i, layer_params in enumerate(params["convolution_layers"]):
            stride = layer_params["stride"]
            window = layer_params["window"]
            mo = layer_params["depth"]
            apply_batch_norm = layer_params["apply_batch_norm"]
            activation = layer_params["activation"]

            output_shape = [self.BATCH_SIZE, dims[i+1], dims[i+1], mo]

            self.g_convolutional_layers.append(
                DeconvolutioinalLayer(
                   mi, mo, output_shape, apply_batch_norm, 
                   window, stride, activation)
                )

            mi = mo

    def generate(self):
        Z = self.g_latent_vars

        for layer in self.g_dense_layers:
            Z = layer.forward(Z)

        for layer in self.g_convolutional_layers:
            Z = layer.forward(Z)

        return Z

    def fit(self, X):
        self.generated_image = self.generate()
        self.session.run(self.generated_image)

    def set_session(self, session):
        self.session = session 


def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def weight_variable(shape, stddev=.1):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(init, name="W")


def bias_variable(shape, val=.1):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init, name="b")


if __name__ == "__main__":
    plt.ion()

    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D, C = Xtrain.shape

    generator_params = {
            "latent_vars": 100,
            "projection": 128,
            "convolution_layers":[
                {"depth":128, "stride":2, "window":5, 
                    "apply_batch_norm":True, "activation":tf.nn.sigmoid},
                {"depth":C, "stride":2, "window":5, 
                    "apply_batch_norm":True, "activation":tf.nn.sigmoid},
                ],
            "fullyconnected_layers":[1024],
            "activation":tf.nn.sigmoid
            }

    discriminator_prams = {
            "convolution_layers":[
                {"depth":2, "stride":2, "window":5, "apply_batch_norm":True},
                {"depth":64, "stride":2, "window":5, "apply_batch_norm":True}
                ],
            "fullyconnected_layers":[1024],
            }
    
    with tf.Session() as session:
        dcgan = DCGAN(
            generator_params=generator_params,
            discriminator_params=discriminator_prams, 
            image_size=28)

        session.run(tf.global_variables_initializer())
        dcgan.set_session(session)
        dcgan.fit(Xtrain)




