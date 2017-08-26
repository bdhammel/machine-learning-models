import tensorflow as tf
import numpy as np

from layers import FullyConnectedLayer
from util import y2indicator, accuracy


class ANN:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.Xin = tf.placeholder(tf.float32, [None, input_size], name="X")
        self.labels = tf.placeholder(tf.float32, [None, output_size], name="labels")
        print(hidden_layer_sizes)

        self.hidden_layers = []

        M1 = input_size
        for i, M2 in enumerate(hidden_layer_sizes):
            self.hidden_layers.append(
                FullyConnectedLayer([M1, M2], "layer_{}".format(i)))
            M1 = M2

        self.output_layer = FullyConnectedLayer([M2, output_size], "output_layer")

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

    def fit(self, X, Y, epochs=10, batch_size=100, test_size=1000):
        
        N, D = X.shape
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

                    c, p = self.session.run([self.cost, self.predictions], feed_dict={
                        self.Xin: X[testbatch_grp], 
                        self.labels: Y[testbatch_grp]})

                    costs.append(c)
                    a = accuracy(Y_flat[testbatch_grp], p)
                    print("i:", i, "j:", j, "nb:", batch_count, "cost:", c, "accuracy:", a)

    def forward(self, X):
        Z = X

        for layer in self.hidden_layers:
            Z = layer.forward(Z)

        return self.output_layer.forward_without_activation(Z)

    def predict(self, X):
        h = self.forward(X)
        return tf.argmax(h, 1)

    def score(self, X, Y):
        p = self.session.run(prediction, feed_dict={self.Xin: X})
        return np.mean(Y == p)





