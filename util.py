import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def accuracy(Y_flat, p):
    return np.mean(Y_flat == p)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def get_data():
    data = pd.read_csv("/Users/bdhammel/Documents/programming/machine_learning/kaggle/mnist/data/train.csv")
    # save the labels to a Pandas series target targets = data['label'].as_matrix()
    # Drop the label feature
    data = data.drop("label",axis=1).as_matrix()
    data = data / data.max()

    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.33, random_state=42)

    return data_train, data_test, targets_train, targets_test

def get_pickled_data():
    """Load the mnist binary because it's faster
    data is already normalized
    """
    print("Loading data")
    d = np.load("/Users/bdhammel/Documents/programming/machine_learning/kaggle/mnist/data/pickled_mnist")
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
    data_train, data_test, targets_train, targets_test = get_pickled_data()
    return (data_train.reshape((len(data_train), 28, 28, 1)), 
            data_test.reshape((len(data_test), 28, 28, 1)), 
            targets_train, 
            targets_test)

def get_image_data():
    data_train, data_test, targets_train, targets_test = get_data()
    return (data_train.reshape((len(data_train), 28, 28, 1)), 
            data_test.reshape((len(data_test), 28, 28, 1)), 
            targets_train, 
            targets_test)
