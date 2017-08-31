import numpy as np
from sklearn.model_selection import train_test_split

def accuracy(Y_flat, p):
    return np.mean(Y_flat == p)

def y2indicator(y):
    """Convert Mnist classes to one hot encoded matrix
    taken from https://github.com/lazyprogrammer
    """
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def get_pickled_data():
    """Load the mnist binary because it's faster
    data is already normalized
    """
    print("Loading data")
    d = np.load("./mnist_data/pickled_mnist")
    print("...done")
    targets = d[:,0]
    data = d[:,1:]
    data = data/255
    assert data.max() <= 1
    assert data.min() >= 0
    print("Shuffle and splitting data into train and test sets")
    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.33, random_state=42)
    print("...done")
    return data_train, data_test, targets_train, targets_test

def get_pickled_image_data():
    """Load the mnist binary in image fromat
    """
    data_train, data_test, targets_train, targets_test = get_pickled_data()
    return (data_train.reshape((len(data_train), 28, 28, 1)), 
            data_test.reshape((len(data_test), 28, 28, 1)), 
            targets_train, 
            targets_test)
