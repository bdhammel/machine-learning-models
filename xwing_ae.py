"""An Xwing type autoencoder
I.E. A symmetric stacked autoencoder

Plotting routine taken from https://github.com/lazyprogrammer
"""
import tensorflow as tf
from util import get_pickled_data
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoderLayer:
    """Shared-weights autoencoder
    weights for decoding are transpose of encoding weights
    """

    def __init__(self, M1, M2, layer_id):
        self._W = tf.Variable(tf.truncated_normal([M1, M2], stddev=.1), name="W")
        self._eb = tf.Variable(tf.zeros([M2]), name="eb")
        self._db = tf.Variable(tf.zeros([M1]), name="db")
        self.layer_id = layer_id

    def encode(self, Z):
        """Encode the given inputs
        """
        with tf.name_scope("encode_layer_{}".format(self.layer_id)):
            return tf.nn.sigmoid(tf.matmul(Z, self.eW) + self.eb)

    def decode(self, Z):
        """Decode the given inputs
        """
        with tf.name_scope("decode_layer_{}".format(self.layer_id)):
            return tf.nn.sigmoid(tf.matmul(Z, self.dW) + self.db)

    @property
    def eW(self):
        """Encoding weight"""
        return self._W

    @property
    def dW(self):
        """Decoding weight"""
        return tf.transpose(self._W)

    @property
    def eb(self):
        """encoding bias"""
        return self._eb

    @property
    def db(self):
        """Decoding bias"""
        return self._db

    
class AutoEncoder:

    def __init__(self, D, layer_sz, load_existing=False):
        self.load_existing = load_existing

        self.tfX = tf.placeholder(tf.float32, [None, D], name="inputs")

        self.layers = []
        M1 = D

        for l, M2 in enumerate(layer_sz):

            with tf.name_scope("AE_layer_{}".format(l)):
                self.layers.append(AutoEncoderLayer(M1, M2, l))

            M1 = M2

        mask = self.corrupt(self.tfX)
        self.Xish = self.forward(mask)

        with tf.name_scope("Training"):
            self.cost = tf.reduce_mean(tf.square(self.Xish - self.tfX))
            self.train_op = tf.train.AdamOptimizer(.1).minimize(
                    self.cost)

        with tf.name_scope("Summarize"):
            tf.summary.scalar("cost", self.cost)
            self.summary_op = tf.summary.merge_all()

        #self.saver = tf.train.Saver(tf.all_variables())

    def set_session(self, session):
        self.session = session

    def forward(self, Z, l=0):
        """Use recursion to drill down into the Autoencoder network
        Once the final layer has been reached, back up through the network,
        decoding the inputs

        Args
        ---
        Z (tensor) : values to encode/decode
        l (int) : the autoencoder layer
        """
        Z = self.layers[l].encode(Z)

        if l+1 < len(self.layers):
            Z = self.forward(Z, l+1)

        Z = self.layers[l].decode(Z)
        return Z

    def corrupt(self, X):
        """Slightly corrupt the input
        """
        return X

    def train(self, X, epochs=3, batch_sz=100, learning_rate=.1, 
            write_log=False, logs_path="/tmp/tensorflow_logs/"):
        """Train the network to reproduce the correct image from the corrupted 
        one
        """
        if self.load_existing:
            try:
                saver.restore(self.session, "/tmp/ae_network.ckpt")
            except:
                print("no model found, running with out save network")

        self.session.run(tf.global_variables_initializer())

        N, D = X.shape
        batches = N // batch_sz

        for epoch in range(epochs):
            test_batch_choices = np.random.choice(N, 100, replace=False)
            Xtest = X[test_batch_choices]

            for batch in range(batches):
                Xbatch = X[batch*batch_sz:(batch+1)*batch_sz]
                _, c, Xish = self.session.run([self.train_op, self.cost, self.Xish], 
                        feed_dict={self.tfX: Xbatch})

                if write_log:
                    summary = self.session.run(
                            self.summary_op, 
                            feed_dict={self.tfX: Xbatch})
                    self.file_writer.add_summary(
                            summary, 
                            epoch * batches + batch)

                if batch % 20 == 0:
                    print("Epoch: ", epoch, "\tBatch: ", batch, "cost", c)

    def predict(self, X):
        return self.session.run(self.Xish, feed_dict={self.tfX:X})


if __name__ == "__main__":
    plt.ion()

    Xtrain, Xtest, Ytrain, Ytest = get_pickled_data()

    N, D = Xtrain.shape
    ae = AutoEncoder(D, [300])
    
    with tf.Session() as sess:
        ae.set_session(sess)
        ae.train(Xtrain, write_log=False)

        done = False
        while not done:
            i = np.random.choice(len(Xtest))
            x = Xtest[i]
            y = ae.predict([x])
            plt.subplot(1,2,1)
            plt.imshow(x.reshape(28,28), cmap='gray')
            plt.title('Original')

            plt.subplot(1,2,2)
            plt.imshow(y.reshape(28,28), cmap='gray')
            plt.title('Reconstructed')

            plt.show()

            ans = input("Generate another?" )
            if ans and ans[0] in ('n' or 'N'):
                done = True

