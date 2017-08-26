import tensorflow as tf
import numpy as np

from util import y2indicator, accuracy
from layers import ConPoolLayer
from ann import ANN 



class CNN:

    def __init__(self, input_size, convolution_layer_info, 
            fully_connected_layer_sizes, output_size):

        self.conpool_layers = []

        self.Xin = tf.placeholder(tf.float32, [None, *input_size], name="X")
        self.labels = tf.placeholder(
                tf.float32, 
                [None, output_size], 
                name="labels")

        C1 = input_size[-1]
        output_reduction_factor = np.array([0,0])

        for i, layer_info in enumerate(convolution_layer_info):
            C2 = layer_info["layer_depth"]
            pool_size = layer_info["strides"]
            self.conpool_layers.append(
                    ConPoolLayer([5, 5, C1, C2], name="C{}".format(i)))
            C1 = C2 
            output_reduction_factor += pool_size[1:-2]

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

        init = tf.global_variables_initializer()
        self.predictions = self.predict(self.Xin)

    def set_session(self, session):
        self.session = session

    def forward(self, Z):

        for conpool_layer in self.conpool_layers:
            Z = conpool_layer.forward(Z)

        Z_flat = tf.reshape(Z, [-1, self.Cout])

        return self.fully_connected_network.forward(Z_flat)

    def predict(self, X):
        h = self.forward(X)
        return tf.argmax(h, 1)

    def fit(self, X, Y, learning_rate=1e-3, epochs=10, batch_size=100, 
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


