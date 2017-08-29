import tensorflow as tf
import numpy as np

from util import y2indicator, accuracy, get_pickled_image_data
from ann import ANN 


class RNN:
    def __init__(self, M):
        self.Xin = tf.placeholder(
                tf.float32, 
                shape=(batch_sz, T, D), 
                name='X')
        self.labels = tf.placeholder(
                tf.int64, 
                shape=(batch_sz, T), 
                name='labels')

        # initial weights
        # note: Wx, Wh, bh are all part of the RNN unit and will be created
        #       by BasicRNNCell
        Wo = init_weight(M, K).astype(np.float32)
        bo = np.zeros(K, dtype=np.float32)

        # make them tf variables
        self.Wo = tf.Variable(Wo)
        self.bo = tf.Variable(bo)

        self.rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)

        outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

        # outputs are now of size (T, batch_sz, M)
        # so make it (batch_sz, T, M)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T*batch_sz, M))

        # Linear activation, using rnn inner loop last output
        logits = tf.matmul(outputs, self.Wo) + self.bo
        predict_op = tf.argmax(logits, 1)
        labels = tf.reshape(self.labels, (T*batch_sz,))

        cost_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        )
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)

    def set_session(self, session):
        self.session = session


    def fit(self, X, Y, batch_sz=20, learning_rate=10e-1, mu=0.99, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
        N, *D = X.shape 
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

                _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})

                cost += c
                for b in xrange(batch_sz):
                    idx = (b + 1)*T - 1
                    n_correct += (p[idx] == Ybatch[b][-1])
                    if i % 10 == 0:
                        print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)
                    if n_correct == N:
                        print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)
                        costs.append(cost)


if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D = Xtrain.shape
    K = len(set(Ytrain))

    with tf.Session() as session:
        rnn = RNN()
        rnn.set_sesison(session)
        session.run(tf.global_variables_initializer())
        rnn.fit(Xtrain, Ytrain)


