"""Artificial Neural Network 

Basic implementation of a fully connected network applied to MNIST data.

Architecture style inspired by tutorials series from https://github.com/lazyprogrammer
"""
import tensorflow as tf
import numpy as np

from util import y2indicator, accuracy, get_pickled_data

class FullyConnectedLayer:
    def __init__(self, shape, namescope, activation_fn=tf.nn.sigmoid):
        """
        Args
        ----
        shape (list, ints) : [input_size, output_size]
        namescope (str) : name scope of the variables
        activation_fn : the activation function to use 
        """
        self.fn = activation_fn
        self.namescope = namescope

        with tf.name_scope(self.namescope):
            self.W = self.weight_variable(shape)
            self.b = self.bias_variable([shape[-1]])

    def forward(self, Z):
        """Propagate data through layer
        """
        with tf.name_scope(self.namescope):
            return self.fn(tf.matmul(Z, self.W) + self.b)

    def forward_without_activation(self, Z):
        """Propagate data through the layer but do not apply a nonlinear 
        activation function. 

        Used for the final output layer
        """
        with tf.name_scope(self.namescope):
            return tf.matmul(Z, self.W) + self.b

    @staticmethod
    def weight_variable(shape, stddev=.1):
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(init, name="W")

    @staticmethod
    def bias_variable(shape, val=.1):
        init = tf.constant(val, shape=shape)
        return tf.Variable(init, name="b")


class ANN:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """Initialize the entire network

        Args
        ----
        input_size (int): Number of features of the input data
        hidden_layer_sizes ([int]), number of nodes in each hidden layer
        output_size (int): number of nodes on the output layer
        """

        # create place holders for the 
        self.Xin = tf.placeholder(
                tf.float32, 
                [None, input_size], 
                name="X")
        self.labels = tf.placeholder(
                tf.float32, 
                [None, output_size], 
                name="labels")

        # store all of the generated hidden layers in order
        self.hidden_layers = []
        M1 = input_size
        for i, M2 in enumerate(hidden_layer_sizes):
            self.hidden_layers.append(
                FullyConnectedLayer(
                    [M1, M2], 
                    namescope="layer_{}".format(i)))
            M1 = M2

        # construct the output layer
        self.output_layer = FullyConnectedLayer(
                [M2, output_size], 
                namescope="output_layer")

        # Define the predicted values of the network, y_hat
        self.logits = self.forward(self.Xin)

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.labels
            )
        ) 

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        self.predictions = self.predict(self.Xin)

    def set_session(self, session):
        """Set the session the graph should be constructed in
        """
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

    def forward(self, Z):
        """Return the logits of the network

        Propagate "Z" though each layer and output with out the activation function 
        """
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
        return self.output_layer.forward_without_activation(Z)

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
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_data()

    N, D = Xtrain.shape
    K = len(set(Ytrain))

    with tf.Session() as session:
        ann = ANN(D, [300, 200, 100], K)
        session.run(tf.global_variables_initializer())
        ann.set_session(session)
        ann.fit(Xtrain, Ytrain)

        print("Test Accuracy: ", ann.score(Xtest, Ytest))


