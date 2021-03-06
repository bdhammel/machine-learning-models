"""Variational Autoencoder

Some components taken from: https://github.com/lazyprogrammer
"""
import tensorflow as tf
from util import get_pickled_data
import matplotlib.pyplot as plt
import numpy as np
from ann import ANN 

st = tf.contrib.bayesflow.stochastic_tensor
Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli

class VariationalAutoencoder:
    def __init__(self, input_size, hidden_layers, output_size):

        # network input
        self.X = tf.placeholder(
                tf.float32, 
                [None, input_size],
                name='X')

        # encoders and decoder are just fully connected networks
        self.encoder = ANN(input_size, hidden_layers, output_size)
        M = output_size//2
        self.decoder = ANN(M, hidden_layers[::-1], input_size)

        # Construct the sampling distribution form the output of the encoder
        self.encoder_out = self.encoder.forward(self.X)
        self.means = self.encoder_out[:, :M]
        self.stddev = tf.nn.softplus(self.encoder_out[:, M:]) + 1e-6

        with st.value_type(st.SampleValue()):
            self.Z = st.StochasticTensor(
                    Normal(loc=self.means, scale=self.stddev))

        # network output
        self.logits = self.decoder.forward(self.Z)
        self.pX = Bernoulli(logits=self.logits)

        # Prior predictive sample 
        standard_normal = Normal(
            loc=np.zeros(M, dtype=np.float32),
            scale=np.ones(M, dtype=np.float32))

        # initialize cost and training
        kl = tf.reduce_sum(
            tf.contrib.distributions.kl_divergence(
                self.Z.distribution, standard_normal
            ), 1)

        expected_log_likelihood = tf.reduce_sum(
            self.pX.log_prob(self.X), 1)

        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.RMSPropOptimizer(
                learning_rate=0.001).minimize(-self.elbo)

        self.X_hat = self.pX.sample()

    def set_session(self, session):
        """Set the session variable for the fully connected encoder decoder networks
        """
        self.session = session
        self.encoder.set_session(session)
        self.decoder.set_session(session)

    def fit(self, X, epochs=30, batch_sz=64):
        costs = []
        n_batches = len(X) // batch_sz
        print("n_batches:", n_batches)

        for i in range(epochs):
            np.random.shuffle(X)

            for j in range(n_batches):
                batch = X[j*batch_sz:(j+1)*batch_sz]
                _, c, = self.session.run((self.train_op, self.elbo), feed_dict={self.X: batch})

                if j % 100 == 0:
                    print("epoch: ", i, " iter: ", j,  " cost: ", c)

    def encode(self, X):
        """Return the calculated means of and input X
        Args
        ----
        X (numpy array) : flattened MNIST image

        Returns
        -------
        means (numpy array) : mean values of the Gaussian fitted models 
        """
        return self.session.run(
            self.means,
            feed_dict={self.X: X})

    def transform(self, X):
        """Transform input image 
        Args
        ----
        X (numpy array) : flattened MNIST image

        Returns
        -------
        Xhat (numpy array) : predicted MNIST image

        """
        return self.session.run(
            self.X_hat,
            feed_dict={self.X: X})



if __name__ == "__main__":
    plt.ion()

    Xtrain, Xtest, Ytrain, Ytest = get_pickled_data()

    N, D = Xtrain.shape
    
    with tf.Session() as session:
        vae = VariationalAutoencoder(D, [200], 200)
        session.run(tf.global_variables_initializer())
        vae.set_session(session)
        vae.fit(Xtrain)

        done = False
        while not done:
            i = np.random.choice(len(Xtest))
            x = Xtest[i]
            y = vae.transform([x])
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

