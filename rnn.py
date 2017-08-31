import tensorflow as tf
import numpy as np

from util import y2indicator, accuracy, get_pickled_image_data

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell

class FullyConnectedLayerWithMemory:

    def __init__(self, hidden_layer_size, output_size, activation_fn=tf.nn.sigmoid):
        self.fn = activation_fn
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.Wo = self.weight_variable(shape=(hidden_layer_size, output_size))
        self.bo = self.bias_variable(shape=[output_size])

        # Input weights, hidden weights, and hidden biases are created by the
        # rnn unit
        #self.rnn_unit = BasicRNNCell(
        #        num_units=hidden_layer_size, activation=self.fn, reuse=None)

        self.rnn_unit = BasicLSTMCell(
                num_units=hidden_layer_size, activation=self.fn, reuse=None)

    def forward(self, X):

        outputs, states = get_rnn_output(
                self.rnn_unit, X, dtype=tf.float32)

        # outputs are now of size (T, batch_sz, M)
        # so make it (batch_sz, T, M)
        """
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(
            outputs, 
            (-1, self.hidden_layer_size))
        """

        return tf.matmul(outputs[-1], self.Wo) + self.bo

    @staticmethod
    def weight_variable(shape, stddev=.1):
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(init, name="W")

    @staticmethod
    def bias_variable(shape, val=.1):
        init = tf.constant(val, shape=shape)
        return tf.Variable(init, name="b")

class RNN:
    def __init__(self, input_size, chunk_size, hidden_layer_sizes, output_size, batch_size=100):

        self.input_size = input_size
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.Xin = tf.placeholder(
                tf.float32, 
                shape=(batch_size, chunk_size, input_size), 
                name='X')
        self.labels = tf.placeholder(
                tf.int64, 
                shape=(batch_size, output_size), 
                name='labels')

        self.layers = []

        self.layers.append(
                FullyConnectedLayerWithMemory(
                    hidden_layer_sizes, output_size)
                )

        self.logits = self.forward(self.Xin)
        self.predict_op = tf.argmax(self.logits, 1)

        self.cost_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            )

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost_op)

    def set_session(self, session):
        self.session = session

    def forward(self, X):
        x = tf.transpose(X, [1,0,2])
        x = tf.reshape(x, [-1, self.chunk_size])
        Z = tf.split(x, self.input_size)

        for layer in self.layers:
            Z = layer.forward(Z)

        return Z

    def fit(self, X, Y, learning_rate=10e-1, activation=tf.nn.sigmoid, epochs=20):
        N, T, D = Xtrain.shape
        Y_flat = np.copy(Y)
        Y = y2indicator(Y)

        self.f = activation

        batch_count = N//self.batch_size
        costs = []

        for i in range(epochs):
            batch_grp = np.arange(0, self.batch_size)

            for j in range(batch_count):
                Xbatch, Ybatch = X[batch_grp], Y[batch_grp]
                Xbatch = Xbatch.reshape((self.batch_size, self.chunk_size, self.input_size))
                batch_grp += self.batch_size

                session.run(
                        [self.train_op, self.cost_op, self.predict_op], 
                        feed_dict={self.Xin: Xbatch, self.labels: Ybatch})

                if j % 20 == 0:
                    testbatch_grp = np.random.choice(
                            N, self.batch_size, replace=True) 

                    c, p = self.session.run(
                            [self.cost_op, self.predict_op], feed_dict={
                                self.Xin: X[testbatch_grp], 
                                self.labels: Y[testbatch_grp]})

                    a = accuracy(Y_flat[testbatch_grp], p)
                    print("i:", i, "j:", j, "nb:", batch_count, "cost:", c, "accuracy:", a)


    def score(self, X, Y):
        """Get an accuracy of the network for a test set
        """
        p = self.session.run(self.predict_op, feed_dict={self.Xin: X})
        return np.mean(Y == p)


if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    # Remove color channel dimension
    Xtrain = Xtrain[...,0]
    Xtest = Xtest[...,0]

    N, T, D = Xtrain.shape
    K = len(set(Ytrain))

    with tf.Session() as session:
        # 300, 200, 100
        rnn = RNN(D, T, 128, K)
        rnn.set_session(session)
        session.run(tf.global_variables_initializer())
        rnn.fit(Xtrain, Ytrain)

        print("Test Accuracy: ", rnn.score(Xtest[:100], Ytest[:100]))

