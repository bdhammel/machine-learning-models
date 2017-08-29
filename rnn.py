import tensorflow as tf
import numpy as np

from util import y2indicator, accuracy, get_pickled_data
from ann import ANN 

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell

class FullyConnectedLayerWithMemory:
    def __init(self, shape, activation_fn):
        self.f = activation_fn
        self.timestep = timestep

        self.W = self.weight_variable(shape)
        self.b = self.bias_variable([shape[-1]])
        self.rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)

    def forward(self, X):
        x = tf.unstack(x, timesteps, 1)

        outputs, states = get_rnn_output(
                self.rnn_unit, x, dtype=tf.float32)

        # outputs are now of size (T, batch_sz, M)
        # so make it (batch_sz, T, M)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T*batch_sz, M))

        return tf.matmul(outputs, self.W) + self.b

    @staticmethod
    def weight_variable(shape, stddev=.1):
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(init, name="W")

    @staticmethod
    def bias_variable(shape, val=.1):
        init = tf.constant(val, shape=shape)
        return tf.Variable(init, name="b")

class RNN:
    def __init__(self, input_size, timesteps, hidden_layer_sizes, output_size):
        self.Xin = tf.placeholder(
                tf.float32, 
                shape=(None, timesteps, input_size), 
                name='X')
        self.labels = tf.placeholder(
                tf.int64, 
                shape=(None, output_size), 
                name='labels')

        self.layers = []

        logits = self.forward(self.Xin)
        self.predict_op = tf.argmax(logits, 1)

        labels = tf.reshape(self.labels, (T*batch_sz,))

        self.cost_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            )

        self.train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)

    def set_session(self, session):
        self.session = session

    def fit(self, X, Y, batch_sz=20, learning_rate=10e-1, mu=0.99, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
        N, D = X.shape 
        Y_flat = np.copy(Y)
        Y = y2indicator(Y)

        M = self.M
        self.f = activation

        costs = []
        n_batches = N // batch_sz

        for i in xrange(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in xrange(n_batches):
                Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

                _, c, p = session.run(
                        [self.train_op, self.cost_op, self.predict_op], 
                        feed_dict={tfX: Xbatch, tfY: Ybatch})

                cost += c

                if i % 10 == 0:
                    print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)


if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_data()

    N, D = Xtrain.shape
    K = len(set(Ytrain))

    with tf.Session() as session:
        rnn = RNN()
        rnn.set_sesison(session)
        session.run(tf.global_variables_initializer())
        rnn.fit(Xtrain, Ytrain)


