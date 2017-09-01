"""Convolutional Autoencoder
"""
import tensorflow as tf
from util import get_pickled_image_data
import matplotlib.pyplot as plt
import numpy as np


class ConvolutionalAELayer:
    """Combination Convolutional and deconvolution layer
    """
    BATCH_SIZE = 100

    def __init__(self, input_dims, layer_info):

        self.input_dims = input_dims
        self.namescope = layer_info["namescope"]
        self.window_shape = layer_info["window_shape"]
        self.window_depth = layer_info["window_depth"]
        self.window_strides = layer_info["window_strides"]

        shape = [*self.window_shape, self.input_dims[-1], self.window_depth]

        with tf.name_scope(self.namescope):
            self.W = self.weight_variable(shape)
            self.b = self.bias_variable([shape[-1]])
            #self.bo = self.bias_variable([shape[-1]])

    def encode(self, Z):
        """Propagate features through Convolutional layer
        """

        with tf.name_scope(self.namescope):
            h_conv = tf.nn.relu(
                        tf.nn.conv2d(
                            Z, self.W, strides=self.window_strides, padding='SAME') 
                        + self.b)

        return h_conv

    def decode(self, Z):
        shape4D = [100, *self.input_dims]  

        with tf.name_scope(self.namescope):
            h_deconv = tf.nn.relu(
                        tf.nn.conv2d_transpose(
                            Z, 
                            self.W, 
                            output_shape=shape4D,
                            strides=self.window_strides, 
                            padding='SAME') 
                        )

        return h_deconv

    @staticmethod
    def weight_variable(shape, stddev=.1):
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(init, name="W")

    @staticmethod
    def bias_variable(shape, val=.1):
        init = tf.constant(val, shape=shape)
        return tf.Variable(init, name="b")

class CAE:

    BATCH_SIZE = 100

    def __init__(self, input_size, layer_info):
        self.X = tf.placeholder(
                    tf.float32, 
                    shape=(None, *input_size), 
                    name='X')

        self.layer = ConvolutionalAELayer(input_size, layer_info)

        self.Xish = self.forward(self.X)
        self.cost = tf.reduce_mean(tf.square(self.Xish - self.X))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

    def train(self, X, epochs=1):
        """Train the network to reproduce the correct image from the corrupted 
        one
        """

        N, *D = X.shape
        batch_count = N // self.BATCH_SIZE

        for epoch in range(epochs):
            test_batch_choices = np.random.choice(N, 100, replace=False)
            Xtest = X[test_batch_choices]

            for batch in range(batch_count):
                Xbatch = X[batch*self.BATCH_SIZE:(batch+1)*self.BATCH_SIZE]
                _, c, Xish = self.session.run([self.train_op, self.cost, self.Xish], 
                        feed_dict={self.X: Xbatch})


                if batch % 20 == 0:
                    print("Epoch: ", epoch, "\tBatch: ", batch, "cost", c)


    def forward(self, Z):
        Z = self.layer.encode(Z)
        Z = self.layer.decode(Z)
        return Z

    def transform(self, X):
        """
        Args
        ----
        X (2D nparray)
        """
        return self.session.run(self.Xish, feed_dict={self.X:X})

    def set_session(self, session):
        self.session = session



if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D = Xtrain.shape

    conpool_layer = {
                "namescope":"CP1",
                "window_depth":32,
                "window_shape":[5, 5],
                "window_strides":[1, 2, 2, 1],
            }

    with tf.Session() as session:
        cae = CAE(D, conpool_layer)
        session.run(tf.global_variables_initializer())
        cae.set_session(session)
        cae.train(Xtrain)

        done = False
        while not done:
            i = np.random.choice(len(Xtest), size=(100))
            x = Xtest[i]
            xish = cae.transform(x)
            plt.subplot(1,2,1)
            plt.imshow(x[0].reshape(28,28), cmap='gray')
            plt.title('Original')

            plt.subplot(1,2,2)
            plt.imshow(xish[0].reshape(28,28), cmap='gray')
            plt.title('Reconstructed')

            plt.show()

            ans = input("Generate another?" )
            if ans and ans[0] in ('n' or 'N'):
                done = True


