"""Convolutional Autoencoder
"""

class ConvolutionalAELayer:
    """Combination Convolutional and deconvolution layer
    """
    def __init__(self, input_dims, layer_info):

        self.input_dims = input_dims
        self.namescope = layer_info["namescope"]
        self.window_shape = layer_info["window_shape"]
        self.window_depth = layer_info["window_depth"]
        self.window_strides = layer_info["window_strides"]

        shape = window_shape + self.window_depth
        print(shape)

        with tf.name_scope(self.namescope):
            self.W = self.weight_variable(shape)
            self.bi = self.bias_variable([shape[-1]])
            self.bo = self.bias_variable([shape[-1]])

    def encode(self, Z):
        """Propagate features through Convolutional layer
        """

        with tf.name_scope(self.namescope):
            h_conv = tf.nn.relu(
                        tf.nn.conv2d(
                            Z, self.W, strides=self.window_strides, padding='SAME') 
                        + self.bi)

        return h_conv

    def decode(self, Z):
        with tf.name_scope(self.namescope):
            h_deconv = tf.nn.relu(
                        tf.nn.conv2d_transpose(
                            Z, self.W.T, strides=self.window_strides, padding='SAME') 
                        + self.bo)

        return h_deconv


class ConvolutionalAutoEncoder:
    def __init__(self):
        pass

    def train(self, X):
        pass

    def forward(self, Z):
        Z = self.layer.encode(Z)
        Z = self.layer.decode(Z)
        return Z

    def transform(self, X):
        """
        Args
        ----
        X (2D nparray)
        """
        pass

def weight_variable(shape, stddev=.1):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(init, name="W")

def bias_variable(shape, val=.1):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init, name="b")


if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D, _ = Xtrain.shape
    K = len(set(Ytrain))

    conpool_layer = {
                "namescope":"CP1",
                "window_depth":32,
                "window_shape":[5, 5],
                "window_strides":[1, 1, 1, 1],
            }

    with tf.Session() as session:
        cae = CAE(D, conpool_layer)
        session.run(tf.global_variables_initializer())
        cae.set_session(session)
        cae.train(Xtrain)

        done = False
        while not done:
            i = np.random.choice(len(Xtest))
            x = Xtest[i]
            y = cae.predict([x])
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
