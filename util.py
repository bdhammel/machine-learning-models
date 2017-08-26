import numpy as np

def accuracy(Y_flat, p):
    return np.mean(Y_flat == p)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
