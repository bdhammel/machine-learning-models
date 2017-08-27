import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from util import get_pickled_data

plt.close('all')


def custom_pca(X):
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the eigenvalue, eigenvector pair from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse= True)

    eig_vecs = [vec[1] for vec in eig_pairs]

if __name__ == "__main__":

    Xtrain, Xtest, Ytrain, Ytest = get_pickled_data()

    # Invoke SKlearn's PCA method 
    n_components = 90 
    pca = PCA(n_components=n_components).fit(Xtrain) 
    evecs = pca.components_.reshape(n_components, 28, 28) 
    data_train = pca.transform(Xtrain) 

    # Extracting the PCA components ( eignevalues ) 
    eigenvalues = pca.components_.reshape(n_components, 28, 28) 
    evacs = pca.components_ 

    # Plot the first 8 eignenvalues
    # -- this was taken from somewhere, I can't remember 
    n_col = 6
    n_row = 5
    plt.figure(figsize=(8,7))

    for i in range(30):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(evecs[i].reshape(28,28), cmap='jet')

        title_text = 'Eigenvalue ' + str(i + 1)
        plt.title(title_text, size=6.5)
        plt.xticks(())
        plt.yticks(())

    plt.show()

    #data_test = pca.transform(Xtest)

    clf = GaussianNB()

    clf.fit(Xtrain, Ytrain)
    print("Naive Bayes accuracy: ", clf.score(Xtest, Ytest)*100)

    eigenvalues = pca.explained_variance_
    eigenvalues = np.array(sorted(eigenvalues, reverse=True))
    plt.plot(eigenvalues.cumsum())
    plt.show()


