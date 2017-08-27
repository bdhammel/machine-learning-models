"""Simple Convolutional neural Network
"""
import tensorflow as tf
import numpy as np

from util import y2indicator, accuracy, get_pickled_image_data
from ann import ANN 


class ConPoolLayer:
    """Combination Convolutional and Pooling layer
    """
    def __init__(self, C1, C2, layer_info):

        self.namescope = layer_info["namescope"]
        window_shape = layer_info["window_shape"]
        self.window_strides = layer_info["window_strides"]
        self.pooling_strides = layer_info["pooling_strides"]
        self.ksize = layer_info["ksize"]

        shape = window_shape + [C1, C2]

        with tf.name_scope(self.namescope):
            self.W = self.weight_variable(shape)
            self.b = self.bias_variable([shape[-1]])

    def forward(self, Z):
        """Propagate features through Convolutional layer
        """

        with tf.name_scope(self.namescope):
            h_conv = tf.nn.relu(
                        tf.nn.conv2d(
                            Z, self.W, strides=[1, 1, 1, 1], padding='SAME') 
                        + self.b)

            h_pool = tf.nn.max_pool(h_conv, ksize=self.ksize, 
                strides=self.pooling_strides, padding='SAME')

        return h_pool

    @staticmethod
    def weight_variable(shape, stddev=.1):
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(init, name="W")

    @staticmethod
    def bias_variable(shape, val=.1):
        init = tf.constant(val, shape=shape)
        return tf.Variable(init, name="b")


class CNN:

    def __init__(self, input_size, convolution_layer_info, 
            fully_connected_layer_sizes, output_size):

        self.conpool_layers = []

        self.Xin = tf.placeholder(
                tf.float32, 
                [None, *input_size], 
                name="X")
        self.labels = tf.placeholder(
                tf.float32, 
                [None, output_size], 
                name="labels")

        C1 = input_size[-1]
        output_reduction_factor = np.array([0,0])

        # Construct the conpool layers
        for i, layer_info in enumerate(convolution_layer_info):
            C2 = layer_info["window_depth"]
            self.conpool_layers.append(
                    ConPoolLayer(C1, C2, layer_info))
            C1 = C2 
            output_reduction_factor += layer_info["pooling_strides"][1:-2]

        # Calculate the input size of the fully connected layer so that it 
        # couples to the convolutional layer
        #   - this will depend on stride of the pooling layer
        self.Cout = int(C2*np.prod(input_size[:-2]/output_reduction_factor))
        self.fully_connected_network = ANN(
                self.Cout, fully_connected_layer_sizes, output_size)

        self.logits = self.forward(self.Xin)

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.labels
            )
        ) 

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        self.predictions = self.predict(self.Xin)

    def fit(self, X, Y, learning_rate=1e-3, epochs=2, batch_size=100, 
            test_size=1000):

        N, *D = X.shape

        Y_flat = np.copy(Y)
        Y = y2indicator(Y)

        batch_count = N//batch_size
        costs = []
        for i in range(epochs):
            batch_grp = np.arange(0, batch_size)
            for j in range(batch_count):

                Xbatch, Ybatch = X[batch_grp], Y[batch_grp]
                batch_grp += batch_size

                self.session.run(
                        [self.train_op, self.cost],
                        feed_dict={self.Xin:Xbatch, self.labels:Ybatch})

                if j % 20 == 0:
                    testbatch_grp = np.random.choice(
                            N, test_size, replace=True) 

                    c, p = self.session.run(
                            [self.cost, self.predictions], feed_dict={
                                self.Xin: X[testbatch_grp], 
                                self.labels: Y[testbatch_grp]})

                    costs.append(c)
                    a = accuracy(Y_flat[testbatch_grp], p)
                    print("i:", i, "j:", j, "nb:", batch_count, "cost:", c, "accuracy:", a)

    def set_session(self, session):
        """Set the session the graph should be constructed in
        """
        self.session = session

    def forward(self, Z):
        """Move input image through entire network 
        """

        for conpool_layer in self.conpool_layers:
            Z = conpool_layer.forward(Z)

        Z_flat = tf.reshape(Z, [-1, self.Cout])

        return self.fully_connected_network.forward(Z_flat)

    def predict(self, X):
        """Find the index of the highest predicted probability 
        (along axis 1)
        """
        h = self.forward(X)
        return tf.argmax(h, 1)

    def score(self, X, Y):
        """Get an accuracy of the network for a test set
        """
        p = self.session.run(self.predictions, feed_dict={self.Xin: X})
        return np.mean(Y == p)


if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D = Xtrain.shape
    K = len(set(Ytrain))

    conpool_layers = [
            {
                "namescope":"CP1",
                "window_depth":32,
                "window_shape":[5, 5],
                "window_strides":[1, 1, 1, 1],
                "pooling_strides":[1, 2, 2, 1],
                "ksize":[1, 2, 2, 1]
            },
            {
                "namescope":"CP2",
                "window_depth":64,
                "window_shape":[5, 5],
                "window_strides":[1, 1, 1, 1],
                "pooling_strides":[1, 2, 2, 1],
                "ksize":[1, 2, 2, 1]
            }]

    with tf.Session() as session:
        cnn = CNN(D, conpool_layers, [1024, 500, 100], K)
        session.run(tf.global_variables_initializer())
        cnn.set_session(session)
        cnn.fit(Xtrain, Ytrain)

        print("Test Accuracy: ", cnn.score(Xtest, Ytest))

