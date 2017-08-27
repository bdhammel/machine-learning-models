"""Convolutional Autoencoder
"""

class ConPoolLayer:
    """Combination Convolutional and Pooling layer
    """
    def __init__(self, C1, C2, layer_info):

        self.namescope = layer_info["namescope"]
        window_shape = layer_info["window_shape"]
        self.window_strides = layer_info["window_strides"]
        self.pooling_strides = layer_info["pooling_strides"]
        self.ksize = layer_info["ksize"]

        shape = window_shape + [C1, C2]

        with tf.name_scope(self.namescope):
            self.W = self.weight_variable(shape)
            self.b = self.bias_variable([shape[-1]])

    def forward(self, Z):
        """Propagate features through Convolutional layer
        """

        with tf.name_scope(self.namescope):
            h_conv = tf.nn.relu(
                        tf.nn.conv2d(
                            Z, self.W, strides=[1, 1, 1, 1], padding='SAME') 
                        + self.b)

            h_pool = tf.nn.max_pool(h_conv, ksize=self.ksize, 
                strides=self.pooling_strides, padding='SAME')

        return h_pool


class DeConPoolLayer:
    def __init__(self, C1, C2, layer_info):

        self.namescope = layer_info["namescope"]
        self.window_strides = layer_info["window_strides"]
        self.output_shape = C2

        shape = window_shape + [C1, C2]

        with tf.name_scope(self.namescope):
            self.W = self.weight_variable(shape)
            self.b = self.bias_variable([shape[-1]])

    def forward(self, Z):
        """Propagate features through Convolutional layer
        """

        with tf.name_scope(self.namescope):
            h_deconv = tf.nn.relu(
                        tf.nn.conv2d_transpose(
                            Z, self.W, 
                            output_shape=self.output_shape,
                            strides=self.window_strides, 
                            padding='SAME') 
                        + self.b)

        return h_deconv


class ConvolutionalAutoEncoder:
    pass


def weight_variable(shape, stddev=.1):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(init, name="W")

def bias_variable(shape, val=.1):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init, name="b")



if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_pickled_image_data()

    N, *D = Xtrain.shape
    K = len(set(Ytrain))

    conpool_layers = [
            {
                "namescope":"CP1",
                "window_depth":32,
                "window_shape":[5, 5],
                "window_strides":[1, 1, 1, 1],
                "pooling_strides":[1, 2, 2, 1],
                "ksize":[1, 2, 2, 1]
            },
            {
                "namescope":"CP2",
                "window_depth":64,
                "window_shape":[5, 5],
                "window_strides":[1, 1, 1, 1],
                "pooling_strides":[1, 2, 2, 1],
                "ksize":[1, 2, 2, 1]
            }]

    with tf.Session() as session:
        cnn = CNN(D, conpool_layers, [1024, 500, 100], K)
        session.run(tf.global_variables_initializer())
        cnn.set_session(session)
        cnn.train(Xtrain)

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
