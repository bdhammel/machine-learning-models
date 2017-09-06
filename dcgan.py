"""Deep Convolutional GAN
"""
import tensorflow as tf
from util import get_pickled_image_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class BaseNNLayer:

    """A basic NN layer containing definitions of general functions
    """

    BATCH_SIZE = 64

    @staticmethod
    def lrelu(x, alpha=0.2):
        """Leaky Relu activation function
        """
        return tf.maximum(alpha*x, x)

    @staticmethod
    def weight_variable(shape, stddev=.1):
        """Construct weight variable of a given size
        """
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        return tf.Variable(init, name="W")

    @staticmethod
    def bias_variable(shape, val=.1):
        init = tf.constant(val, shape=shape)
        return tf.Variable(init, name="b")


class FullyConnectedLayer(BaseNNLayer):
    def __init__(self, shape, activation_fn=tf.nn.sigmoid):
        """
        Args
        ----
        shape (list, ints) : [input_size, output_size]
        activation_fn : the activation function to use 
        """
        self.fn = activation_fn

        self.W = self.weight_variable(shape)
        self.b = self.bias_variable([shape[-1]])

    def forward(self, Z):
        """Propagate data through layer
        """
        return self.fn(tf.matmul(Z, self.W) + self.b)

    def train(self):
        pass


class ConvolutionalLayer(BaseNNLayer):

    def __init__(self, filter_size, stride):
        self.stride = [1, stride, stride, 1]

        self.W = self.weight_variable(filter_size)
        self.b = self.bias_variable([filter_size[-1]])

        # Build operations for pretraining

        self.conv_X = tf.placeholder()
        Z = self.forward(self.conv_X)
        
        self.Xish = tf.nn.relu(
                    tf.nn.conv2d_transpose(
                        Z, 
                        self.W, 
                        output_shape=shape4D,
                        strides=self.window_strides, 
                        padding='SAME') 
                    )


        self.cost = tf.reduce_mean(tf.square(self.Xish - self.conv_X))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

    def forward(self, Z):
        """Propagate features through Convolutional layer
        """

        h_conv = tf.nn.relu(
                    tf.nn.conv2d(
                        Z, self.W, strides=self.stride, padding='SAME') 
                    + self.b)

        return h_conv

    def train(self, X):
        """Pre train the convolution layer
        Build a 1 layer deep convolutional auto encoder
        """

        N, *D = X.shape
        batch_count = N // self.BATCH_SIZE



        for epoch in range(epochs):
            test_batch_choices = np.random.choice(N, 100, replace=False)
            Xtest = X[test_batch_choices]

            for batch in range(batch_count):
                Xbatch = X[batch*self.BATCH_SIZE:(batch+1)*self.BATCH_SIZE]
                _, c, Xish = self.session.run([self.train_op, self.cost, self.Xish], 
                        feed_dict={tfX: Xbatch})


                if batch % 20 == 0:
                    print("Epoch: ", epoch, "\tBatch: ", batch, "cost", c)



class DeconvolutionalLayer(BaseNNLayer):
    """Combination Convolutional and deconvolution layer
    """

    def __init__(self, filter_size, stride, output_dims):

        self.output_dims = output_dims
        self.stride = [1, stride, stride, 1]

        self.W = self.weight_variable(filter_size)

    def forward(self, Z):
        shape = [self.BATCH_SIZE, *self.output_dims]  

        h_deconv = tf.nn.sigmoid(
                    tf.nn.conv2d_transpose(
                        Z, 
                        self.W, 
                        output_shape=shape,
                        strides=self.stride,
                        padding='SAME') 
                    )

        return h_deconv


class Generator:

    BATCH_SIZE = 64

    def __init__(self, g_info):

        self.Z = tf.Variable(tf.random_uniform(
                shape=(self.BATCH_SIZE, *g_info["latent_var_size"]), 
                minval=0, 
                maxval=1), name="latent_var")

        self.X = tf.placeholder(
                dtype=tf.float32,
                shape=[self.BATCH_SIZE, 28, 28, 1],
                name="X")

        dims = self._calculate_layer_dims(g_info)

        M1 = g_info["latent_var_size"][0]
        self.dense_layers = []
        for M2 in self.dense_dims:
            self.dense_layers.append(FullyConnectedLayer((M1, M2)))
            M1 = M2

        self.conv_layers = []
        C1 = self.conv_dims[0][-1]
        for i, layer_info in enumerate(g_info["conv_layers"]):
            C2, filter_size, stride, _ = layer_info
            self.conv_layers.append(
                    DeconvolutionalLayer(
                        filter_size=[filter_size, filter_size, C2, C1], 
                        stride=stride,
                        output_dims=self.conv_dims[i+1]
                        )
                    )
            C1 = C2

        self.Xish = self.generate()
        self.cost = tf.reduce_mean(tf.square(self.Xish - self.X))
        self.pretrain_op = tf.train.AdamOptimizer().minimize(self.cost)

    def _calculate_layer_dims(self, g_info):
        """
        conv dims are of form (side, side, depth)
        """
        L1 = g_info["output_dims"]

        self.conv_dims = [L1]
        for layer in reversed(g_info["conv_layers"]):
            depth, _, stride, _ = layer
            L2 = [*np.divide(L1[:-1], stride).astype(int), None]
            self.conv_dims[-1][-1] = depth
            self.conv_dims.append(L2)
            L1 = L2

        self.conv_dims[-1][-1] = g_info["projection"]
        self.conv_dims = self.conv_dims[::-1]

        self.dense_dims = []

        M1 = np.prod(L1)
        self.dense_dims.append(M1)

        for M2, _ in reversed(g_info["dense_layers"]):
            self.dense_dims.append(M2)
            M2 = M1


        self.dense_dims = self.dense_dims[::-1]

    def set_session(self, session):
        self.session = session 

    def generate(self):

        z = self.Z
        for layer in self.dense_layers:
            z = layer.forward(z)

        z = tf.reshape(z, shape=[-1, *self.conv_dims[0]])

        for layer in self.conv_layers:
            z = layer.forward(z)

        return z

    def generate_sample(self):
        return self.session.run(self.Xish)

    def train(self, X, training_epoch=200):

        # Invoke SKlearn's PCA method 
        n_components = 90 
        N, *D = X.shape
        Xflat = np.reshape(X, (N, np.prod(D)))
        pca = PCA(n_components=n_components).fit(Xflat) 
        evecs = pca.components_.reshape(n_components, 28, 28, 1) 
        e0 = evecs[0]
        Xtrain = np.abs(e0)
        Xtrain /= Xtrain.max()

        for i in range(training_epoch):
            Xsample = Xtrain*np.random.random(size=(self.BATCH_SIZE, 28, 28, 1))
            _, c = self.session.run(
                    [self.pretrain_op, self.cost], 
                    feed_dict={self.X: Xsample})

            if i % 10 == 0:
                print("i: ", i, "c: ", c)

        return Xtrain


class Discriminator:

    BATCH_SIZE = 64

    def __init__(self, d_info):

        self.Xin = tf.placeholder(
                tf.float32, 
                (self.BATCH_SIZE, *d_info["input_dims"]), 
                name="X")

        self.labels = tf.placeholder(
                tf.float32,
                (self.BATCH_SIZE, 1),
                name="labels")


        self._calculate_layer_dims(d_info)

        self.conv_layers = []
        C1 = d_info["input_dims"][-1]
        for i, layer_info in enumerate(d_info["conv_layers"]):
            C2, filter_size, stride, _ = layer_info
            self.conv_layers.append(
                    ConvolutionalLayer(
                        filter_size=[filter_size, filter_size, C1, C2], 
                        stride=stride,
                        )
                    )
            C1 = C2

        self.dense_layers = []
        M1 = self.dense_dims[0]

        for M2 in self.dense_dims[1:]:
            self.dense_layers.append(
                FullyConnectedLayer((M1, M2))
            )
            M1 = M2

        self.output_layer = FullyConnectedLayer((M1, 1))

    def _calculate_layer_dims(self, l_info):
        """
        conv dims are of form (side, side, depth)
        """
        L1 = l_info["input_dims"]

        self.conv_dims = [L1]
        for layer in l_info["conv_layers"]:
            depth, _, stride, _ = layer
            L2 = [*np.divide(L1[:-1], stride).astype(int), depth]
            self.conv_dims.append(L2)
            L1 = L2

        self.dense_dims = []

        M1 = np.prod(L1)
        self.dense_dims.append(M1)

        for M2 in l_info["dense_layers"]:
            self.dense_dims.append(M2)
            M2 = M1

    def set_session(self, session):
        self.session = session

    def forward(self, Z):

        for layer in self.conv_layers:
            Z = layer.forward(Z)

        Z = tf.reshape(Z, [-1, self.dense_dims[0]])

        for layer in self.dense_layers:
            Z = layer.forward(Z)

        return tf.matmul(Z, self.output_layer.W) + self.output_layer.b


class DCGAN:

    BATCH_SIZE = 100

    def __init__(self, discriminator_params, generator_params):

        with tf.name_scope("Generator"):
            self.generator = Generator(generator_params)
            sample = self.generator.generate()

        with tf.name_scope("Discriminator"):
            self.discriminator = Discriminator(discriminator_params)
            logits = self.discriminator.forward(self.Xin)
            sample_logits = self.discriminator.forward(sample)

        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits,
          labels=tf.ones_like(logits)
        )

        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=sample_logits,
          labels=tf.zeros_like(sample_logits)
        )

        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)

        self.g_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sample_logits,
                labels=tf.ones_like(sample_logits)
                )
            )

        real_predictions = tf.cast(logits > 0, tf.float32)
        fake_predictions = tf.cast(sample_logits < 0, tf.float32)

        num_predictions = 2.0*self.BATCH_SIZE
        num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
        self.d_accuracy = num_correct / num_predictions

        self.d_train_op = tf.train.AdamOptimizer(0.0002).minimize(self.d_cost)
        self.g_train_op = tf.train.AdamOptimizer(0.0002).minimize(self.g_cost)


    def fit(self, X, epochs=1):

        N = len(X)
        n_batches = N // self.BATCH_SIZE
        total_iters = 0

        for i in range(epochs):

            for j in range(n_batches):

                batch = X[j*self.BATCH_SIZE:(j+1)*self.BATCH_SIZE]

                _, d_cost, d_acc = self.session.run(
                    (self.d_train_op, self.d_cost, self.d_accuracy),
                    feed_dict={self.Xin: batch},
                )

                # train the generator
                _, g_cost1 = self.session.run(
                    (self.g_train_op, self.g_cost)
                )

                # g_costs.append(g_cost1)
                _, g_cost2 = self.session.run(
                    (self.g_train_op, self.g_cost)
                )

                if j % 10 == 0:
                    print(i, j)

                # print("  batch: %d/%d  -  dt: %s - d_acc: %.2f" % (j+1, n_batches, datetime.now() - t0, d_acc))


    def set_session(self, session):
        self.session = session 
        self.discriminator.set_session(session)
        self.generator.set_session(session)

    @property
    def Xin(self):
        return self.discriminator.Xin


if __name__ == "__main__":

    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D, C = Xtrain.shape

    generator_params = {
        'latent_var_size': [100],
        'projection': 128,
        'bn_after_project': False,
        'conv_layers': [(128, 5, 2, True), (C, 5, 2, False)],
        'dense_layers': [(1024, True)],
        'output_activation': tf.sigmoid,
        'output_dims': [*D, C]
    }

    discriminator_prams = {
        "input_dims":[*D, C],
        "conv_layers":[(2, 5, 2, True), (64, 5, 2, True)],
        "dense_layers":[1024],
        'output_dims': [1]
    }
    
    with tf.Session() as session:
        dcgan = DCGAN(
            generator_params=generator_params,
            discriminator_params=discriminator_prams, 
            )

        session.run(tf.global_variables_initializer())
        dcgan.set_session(session)
        dcgan.generator.train(Xtrain)
        #dcgan.fit(Xtrain)
        imgs = dcgan.generator.generate_sample()
        print(imgs.shape)

        for img in imgs:
            plt.imshow(imgs[0,...,0])
            plt.show()

            if input("Continue? > ") == "n":
                break




